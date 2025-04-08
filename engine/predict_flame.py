# Copyright 2024-2025 The Alibaba 3DAIGC Team Authors. All rights reserved.
import json
import sys

sys.path.append("./")
sys.path.append("./gaga_track")
import os
import shutil
import traceback
import warnings
from os.path import join

from gaga_track.track_video import Tracker
from tqdm.std import TqdmExperimentalWarning

warnings.simplefilter("ignore", category=UserWarning, lineno=0, append=False)
warnings.simplefilter("ignore", category=TqdmExperimentalWarning, lineno=0, append=True)


def init_gaga_track(model_path, device):
    return Tracker(focal_length=12.0, model_path=model_path, device=device)


def save_flame_to_json(flame_results, out_path):
    for k, v in flame_results.items():
        frame_name = int(k.split("_")[-1]) + 1
        json_path = join(out_path, f"{frame_name:05d}.json")
        flame_param = {
            "bbox": v["bbox"].tolist(),
            "frame_bbox": v["frame_bbox"].tolist(),
            "shapecode": v["shapecode"].tolist(),
            "expcode": v["expcode"].tolist(),
            "posecode": v["posecode"].tolist(),
            "neckcode": v["neckcode"].tolist(),
            "eyecode": v["eyecode"].reshape(-1).tolist(),
            "transform_matrix": v["transform_matrix"].reshape(-1).tolist(),
        }
        with open(json_path, "w", encoding="utf-8") as fp:
            json.dump(flame_param, fp)


def estimate_flame(gagatrack, video_dir):

    output_path = join(video_dir, "flame_params")
    if os.path.exists(output_path) and os.path.isdir(output_path):
        shutil.rmtree(output_path)

    os.makedirs(output_path, exist_ok=True)

    try:
        optim_results = gagatrack.track_video(video_dir, output_path)

        if len(optim_results) > 0:
            save_flame_to_json(optim_results, output_path)

    except:
        traceback.print_exc()
