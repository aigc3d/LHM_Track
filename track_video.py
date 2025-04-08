# Copyright 2024-2025 The Alibaba 3DAIGC Team Authors. All rights reserved.

import argparse
import os
import traceback

import cv2
import torch

from engine.pose_estimation.video2motion import Video2MotionPipeline
from engine.predict_box import init_box_model, predict_box
from engine.predict_flame import estimate_flame, init_gaga_track
from engine.predict_samurai import run_samurai
from engine.predict_sapiens_pose import run_sapiens


def extract_frame(video_path, out_path):
    os.makedirs(out_path, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"fail to load video file {video_path}"
    fps = cap.get(cv2.CAP_PROP_FPS)
    fid = 0
    while cap.isOpened():
        flag, frame = cap.read()
        if not flag:
            break
        fid += 1
        cv2.imwrite(os.path.join(out_path, f"{fid:05d}.png"), frame)
    return fps


class VideoTracker:

    def __init__(self, model_path, device, opt):
        self.device = device
        self.model_path = model_path
        self.opt = opt
        self._init_models(model_path)

    def _init_models(self, model_path):

        self.sam2seg = init_box_model(model_path)

        self.video2motion = Video2MotionPipeline(
            os.path.join(model_path, "human_model_files"),
            device=self.device,
            kp_mode="sapiens",
            track_mode="samurai",
            is_smooth=False,
            pad_ratio=self.opt.pad_ratio,
        )
        self.gaga_track = init_gaga_track(
            os.path.join(model_path, "gagatracker"), self.device
        )

    def __call__(self, input_path, output_path):
        os.makedirs(output_path, exist_ok=True)
        if os.path.isdir(input_path):
            self.process_folder(input_path, output_path)
        else:
            self.process_video(input_path, output_path)

    def process_folder(self, folder_path, output_path):
        video_names = [
            video_name
            for video_name in sorted(os.listdir(folder_path))
            if video_name.endswith(".mp4")
        ]
        for video_name in video_names:
            video_path = os.path.join(folder_path, video_name)
            try:
                self.process_video(video_path, output_path)
            except:
                traceback.print_exc()

    def process_video(self, video_path, output_dir):
        if not video_path.endswith(".mp4"):
            print(f"{video_path} is not a video file")
            return

        print(f"Processing video: {video_path}")
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(output_dir, video_name)
        os.makedirs(output_path, exist_ok=True)

        # 1. extract frames
        frame_path = os.path.join(output_path, "imgs_png")
        fps = extract_frame(video_path, frame_path)

        # 2. predict human bbox for first frame
        predict_box(self.sam2seg, output_path)

        # 3. human tracking and segmentation
        run_samurai(self.model_path, output_path)

        # 4. predict 2D keypoints
        run_sapiens(self.model_path, output_path)

        # 5. predict smplx
        self.video2motion(output_path, output_dir, fps)

        # 6. predict flame
        estimate_flame(self.gaga_track, output_path)
        print(f"Finish processing video: {video_path}")


def get_parse():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--video_path",
        type=str,
        required=True,
        help="a mp4 file path or a folder containing mp4 files",
    )
    parser.add_argument(
        "--output_path", type=str, default="./train_data/custom_dataset"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./pretrained_models",
        help="model_path",
    )
    parser.add_argument(
        "--pad_ratio",
        type=float,
        default=0.2,
        help="padding images for more accurate estimation results",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    opt = get_parse()
    assert (
        torch.cuda.is_available()
    ), "CUDA is not available, please check your environment"
    assert os.path.exists(opt.video_path), "The video is not exists"
    os.makedirs(opt.output_path, exist_ok=True)

    device = "cuda:0"
    tracker = VideoTracker(opt.model_path, device, opt)
    tracker(opt.video_path, opt.output_path)
