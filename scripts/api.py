from typing import List, Optional

import numpy as np
from fastapi import FastAPI, Body
from fastapi.exceptions import HTTPException
from pydantic import BaseModel
import json

from PIL import Image

import gradio as gr

from modules.api.models import *
from modules.api import api
from modules import shared

from scripts import external_code, global_state
from scripts.processor import preprocessor_filters
from scripts.logging import logger
from annotator.openpose import draw_poses, decode_json_as_poses
from annotator.openpose.animalpose import draw_animalposes

from scripts.processor import (
    preprocessor_sliders_config,
    no_control_mode_preprocessors,
    flag_preprocessor_resolution,
    model_free_preprocessors,
    preprocessor_filters,
    HWC3,
)

def encode_to_base64(image):
    if isinstance(image, str):
        return image
    elif isinstance(image, Image.Image):
        return api.encode_pil_to_base64(image)
    elif isinstance(image, np.ndarray):
        return encode_np_to_base64(image)
    else:
        return ""


def encode_np_to_base64(image):
    pil = Image.fromarray(image)
    return api.encode_pil_to_base64(pil)

def singleton(cls):
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance

@singleton
class Preprocessors:
    def __init__(self):
        self.preprocessors = global_state.cache_preprocessors(global_state.cn_preprocessor_modules)

preprocessors = Preprocessors()

def controlnet_api(_: gr.Blocks, app: FastAPI):
    @app.get("/controlnet/version")
    async def version():
        return {"version": external_code.get_api_version()}

    @app.get("/controlnet/model_list")
    async def model_list(update: bool = True):
        up_to_date_model_list = external_code.get_models(update=update)
        logger.debug(up_to_date_model_list)
        return {"model_list": up_to_date_model_list}

    @app.get("/controlnet/module_list")
    async def module_list(alias_names: bool = False):
        _module_list = external_code.get_modules(alias_names)
        logger.debug(_module_list)

        return {
            "module_list": _module_list,
            "module_detail": external_code.get_modules_detail(alias_names),
        }

    @app.get("/controlnet/control_types")
    async def control_types():
        def format_control_type(
            filtered_preprocessor_list,
            filtered_model_list,
            default_option,
            default_model,
        ):
            return {
                "module_list": filtered_preprocessor_list,
                "model_list": filtered_model_list,
                "default_option": default_option,
                "default_model": default_model,
            }

        return {
            "control_types": {
                control_type: format_control_type(
                    *global_state.select_control_type(control_type)
                )
                for control_type in preprocessor_filters.keys()
            }
        }

    @app.get("/controlnet/settings")
    async def settings():
        max_models_num = external_code.get_max_models_num()
        return {"control_net_unit_count": max_models_num}

    cached_cn_preprocessors = global_state.cache_preprocessors(
        global_state.cn_preprocessor_modules
    )

    @app.post("/controlnet/detect")
    async def detect(
        controlnet_module: str = Body("none", title="Controlnet Module"),
        controlnet_input_images: List[str] = Body([], title="Controlnet Input Images"),
        controlnet_processor_res: int = Body(
            512, title="Controlnet Processor Resolution"
        ),
        controlnet_threshold_a: float = Body(64, title="Controlnet Threshold a"),
        controlnet_threshold_b: float = Body(64, title="Controlnet Threshold b"),
    ):
        controlnet_module = global_state.reverse_preprocessor_aliases.get(
            controlnet_module, controlnet_module
        )

        if controlnet_module not in cached_cn_preprocessors:
            raise HTTPException(status_code=422, detail="Module not available")

        if len(controlnet_input_images) == 0:
            raise HTTPException(status_code=422, detail="No image selected")

        logger.info(
            f"Detecting {str(len(controlnet_input_images))} images with the {controlnet_module} module."
        )

        results = []
        poses = []

        processor_module = cached_cn_preprocessors[controlnet_module]

        for input_image in controlnet_input_images:
            img = external_code.to_base64_nparray(input_image)

            class JsonAcceptor:
                def __init__(self) -> None:
                    self.value = None

                def accept(self, json_dict: dict) -> None:
                    self.value = json_dict

            json_acceptor = JsonAcceptor()

            results.append(
                processor_module(
                    img,
                    res=controlnet_processor_res,
                    thr_a=controlnet_threshold_a,
                    thr_b=controlnet_threshold_b,
                    json_pose_callback=json_acceptor.accept,
                )[0]
            )

            if "openpose" in controlnet_module:
                assert json_acceptor.value is not None
                poses.append(json_acceptor.value)

        global_state.cn_preprocessor_unloadable.get(controlnet_module, lambda: None)()
        results64 = list(map(encode_to_base64, results))
        res = {"images": results64, "info": "Success"}
        if poses:
            res["poses"] = poses

        return res

    @app.post("/controlnet/preprocess")
    async def preprocess(
        input_images: dict = Body({}, title="Controlnet Input Images"),
        module: str = Body("none", title="Controlnet Preprocessor Module"),
        pres: int = Body(512, title="Controlnet Preprocessor Resolution"),
        pthr_a: float = Body(-1, title="Controlnet Preprocessor Threshold a"),
        pthr_b: float = Body(-1, title="Controlnet Preprocessor Threshold b"),
        wide: int = Body(512, title="Controlnet Target Image Width"),
        height: int = Body(512, title="Controlnet Target Image Height"),
        pixel_perfect: bool = Body(False, title="Controlnet Pixel Perfect"),
        remove_mask: str = Body("Crop and Resize", title="Controlnet Remove Mask"),
    ):
        input_images = {
            "image": external_code.to_base64_nparray(input_images["image"]),
            "mask": external_code.to_base64_nparray(input_images["mask"]),
        }
        return run_annotator(input_images, module, pres, pthr_a, pthr_b, wide, height, pixel_perfect, remove_mask)

    def run_annotator(image, module, pres, pthr_a, pthr_b, t2i_w, t2i_h, pp, rm):
        if image is None:
            raise HTTPException(status_code=422, detail="No image selected")

        img = HWC3(image["image"])
        has_mask = not (
                (image["mask"][:, :, 0] <= 5).all()
                or (image["mask"][:, :, 0] >= 250).all()
        )
        if "inpaint" in module:
            color = HWC3(image["image"])
            alpha = image["mask"][:, :, 0:1]
            img = np.concatenate([color, alpha], axis=2)
        elif has_mask and not shared.opts.data.get(
                "controlnet_ignore_noninpaint_mask", False
        ):
            img = HWC3(image["mask"][:, :, 0])

        module = global_state.get_module_basename(module)
        preprocessor = cached_cn_preprocessors[module]

        if pp:
            pres = external_code.pixel_perfect_resolution(
                img,
                target_H=t2i_h,
                target_W=t2i_w,
                resize_mode=external_code.resize_mode_from_value(rm),
            )

        class JsonAcceptor:
            def __init__(self) -> None:
                self.value = ""

            def accept(self, json_dict: dict) -> None:
                self.value = json_dict

        json_acceptor = JsonAcceptor()

        result, is_image = preprocessor(
            img,
            res=pres,
            thr_a=pthr_a,
            thr_b=pthr_b,
            json_pose_callback=json_acceptor.accept
            if "openpose" in module
            else None,
        )

        if not is_image:
            result = img

        result = external_code.visualize_inpaint_mask(result)

        pose = None
        if "openpose" in module:
            assert json_acceptor.value is not None
            pose = json_acceptor.value

        return {
            "image": encode_np_to_base64(result),
            "pose": pose,
        }

    class Person(BaseModel):
        pose_keypoints_2d: List[float]
        hand_right_keypoints_2d: Optional[List[float]]
        hand_left_keypoints_2d: Optional[List[float]]
        face_keypoints_2d: Optional[List[float]]

    class PoseData(BaseModel):
        people: List[Person]
        canvas_width: int
        canvas_height: int

    @app.post("/controlnet/render_openpose_json")
    async def render_openpose_json(
        pose_data: List[PoseData] = Body([], title="Pose json files to render.")
    ):
        if not pose_data:
            return {"info": "No pose data detected."}
        else:

            def draw(poses, animals, H, W):
                if poses:
                    assert len(animals) == 0
                    return draw_poses(poses, H, W)
                else:
                    return draw_animalposes(animals, H, W)

            return {
                "images": [
                    encode_to_base64(draw(*decode_json_as_poses(pose.dict())))
                    for pose in pose_data
                ],
                "info": "Success",
            }


try:
    import modules.script_callbacks as script_callbacks

    script_callbacks.on_app_started(controlnet_api)
except:
    pass
