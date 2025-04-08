#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)

import json
import os
import pickle
import random
import shutil
import sys
from glob import glob

import numpy as np
import torch
import torchvision
from tqdm.rich import tqdm

from .emica_encoder import EmicaEncoder, ImageEngine
from .engine_optim import OptimEngine
from .flame_model import FLAMEModel, RenderMesh
from .human_matting import StyleMatteEngine as HumanMattingEngine
from .utils_lmdb import LMDBEngine
from .vgghead_detector import VGGHeadDetector


class CoreEngine:

    def __init__(self, focal_length, model_path, device="cuda"):
        random.seed(42)
        self._device = device
        # paths and data engine
        self.model_path = model_path
        self.emica_encoder = EmicaEncoder(model_path=self.model_path, device=device)
        self.emica_data_engine = ImageEngine(model_path=self.model_path, device=device)
        self.vgghead_encoder = VGGHeadDetector(
            model_path=self.model_path, device=device
        )
        self.matting_engine = HumanMattingEngine(
            model_path=self.model_path, device=device
        )
        calibration_results = {"focal_length": focal_length, "verts_scale": 5.0}
        self.calibration_results = calibration_results
        self.optim_engine = OptimEngine(
            self.model_path, self.calibration_results, device=device
        )

    def build_video(
        self, video_path, output_path, matting=False, background=0.0, if_crop=True
    ):
        def smooth_bbox(all_bbox, alpha=1):
            smoothed_bbox = [
                all_bbox[0]
            ]  # Initialize the smoothed data with the first value of the input data
            for i in range(1, len(all_bbox)):
                smoothed_value = (
                    alpha * all_bbox[i] + (1 - alpha) * smoothed_bbox[i - 1]
                )
                smoothed_bbox.append(smoothed_value)
            return smoothed_bbox

        video_name = os.path.basename(video_path).split(".")[0]

        os.makedirs(output_path, exist_ok=True)
        self.raw_info = {}
        if not os.path.exists(os.path.join(output_path, "img_lmdb")):

            all_image_squence = glob(os.path.join(video_path, "imgs_png", "*.png"))

            all_image_squence = sorted(all_image_squence)

            assert (
                len(all_image_squence) > 0
            ), "No frames in the video, reading video failed."
            print(
                f"Processing video {video_path} with {len(all_image_squence)} frames."
            )
            self.if_crop = if_crop
            if if_crop:
                all_frames_boxes, all_frames_idx = [], []
                frames_data = []
                for fidx, frame_path in tqdm(
                    enumerate(all_image_squence), total=len(all_image_squence)
                ):
                    frame = torchvision.io.read_image(frame_path)
                    frames_data.append(frame)
                    _, bbox, _ = self.vgghead_encoder(frame, fidx, only_vgghead=True)
                    if bbox is not None:
                        all_frames_idx.append(fidx)
                        all_frames_boxes.append(bbox.cpu())
                if not len(all_frames_boxes):
                    print(
                        "No face detected in the video: {}, tracking failed.".format(
                            video_path
                        )
                    )
                    return None
                frames_data = [
                    frames_data[i] for i in all_frames_idx
                ]  # frames_data[all_frames_idx]
                all_frames_boxes = smooth_bbox(all_frames_boxes, alpha=0.5)
                lmdb_engine = LMDBEngine(
                    os.path.join(output_path, "img_lmdb"), write=True
                )
                for fidx, frame in tqdm(enumerate(frames_data), total=len(frames_data)):
                    frame_bbox = all_frames_boxes[fidx]
                    frame_bbox = expand_bbox(frame_bbox, scale=1.65).long()
                    crop_frame = torchvision.transforms.functional.crop(
                        frame,
                        top=frame_bbox[1],
                        left=frame_bbox[0],
                        height=frame_bbox[3] - frame_bbox[1],
                        width=frame_bbox[2] - frame_bbox[0],
                    )
                    crop_frame = torchvision.transforms.functional.resize(
                        crop_frame, (512, 512), antialias=True
                    )
                    self.raw_info[f"{video_name}_{fidx}"] = {
                        "image_size": [frame.shape[1], frame.shape[2]],
                        "frame_bbox": frame_bbox,
                    }
                    # frame = torchvision.transforms.functional.center_crop(frame, 512)
                    if matting:
                        crop_frame = (
                            self.matting_engine(
                                crop_frame / 255.0,
                                return_type="matting",
                                background_rgb=background,
                            ).cpu()
                            * 255.0
                        )
                    lmdb_engine.dump(
                        f"{video_name}_{fidx}", payload=crop_frame, type="image"
                    )
                lmdb_engine.random_visualize(os.path.join(output_path, "visualize.jpg"))
                lmdb_engine.close()
            else:
                lmdb_engine = LMDBEngine(
                    os.path.join(output_path, "img_lmdb"), write=True
                )
                for fidx, frame in tqdm(
                    enumerate(frames_data), total=frames_data.shape[0]
                ):

                    self.raw_info[f"{video_name}_{fidx}"] = {
                        "image_size": [frame.shape[1], frame.shape[2]],
                    }
                    frame = torchvision.transforms.functional.resize(
                        frame, 512, antialias=True
                    )
                    frame = torchvision.transforms.functional.center_crop(frame, 512)
                    if matting:
                        frame = (
                            self.matting_engine(
                                frame / 255.0,
                                return_type="matting",
                                background_rgb=background,
                            ).cpu()
                            * 255.0
                        )
                    lmdb_engine.dump(
                        f"{video_name}_{fidx}", payload=frame, type="image"
                    )
                lmdb_engine.random_visualize(os.path.join(output_path, "visualize.jpg"))
                lmdb_engine.close()
            return 30
        else:
            return 30

    def build_image_squences(
        self,
        image_squence_dir,
        output_path,
        matting=False,
        background=0.0,
        if_crop=False,
    ):
        def smooth_bbox(all_bbox, alpha=0.7):
            smoothed_bbox = [
                all_bbox[0]
            ]  # Initialize the smoothed data with the first value of the input data
            for i in range(1, len(all_bbox)):
                smoothed_value = (
                    alpha * all_bbox[i] + (1 - alpha) * smoothed_bbox[i - 1]
                )
                smoothed_bbox.append(smoothed_value)
            return smoothed_bbox

        video_name = os.path.basename(image_squence_dir).split(".")[0]
        self.raw_info = {}
        self.if_crop = if_crop

        lmdb_engine = LMDBEngine(os.path.join(output_path, "img_lmdb"), write=True)
        all_image_squence = glob(os.path.join(image_squence_dir, "images", "*.png"))
        all_image_squence = sorted(all_image_squence)
        os.makedirs(output_path, exist_ok=True)

        for fidx, frame_path in tqdm(
            enumerate(all_image_squence), total=len(all_image_squence)
        ):
            frame = torchvision.io.read_image(frame_path)

            frame_name = frame_path.split("/")[-1]
            self.raw_info[frame_name] = {
                "image_size": [frame.shape[1], frame.shape[2]],
            }
            frame = torchvision.transforms.functional.resize(frame, 512, antialias=True)
            frame = torchvision.transforms.functional.center_crop(frame, 512)

            lmdb_engine.dump(frame_name, payload=frame, type="image")
        lmdb_engine.random_visualize(os.path.join(output_path, "visualize.jpg"))
        lmdb_engine.close()

        return 30

    def track_base(self, lmdb_engine, output_path):
        if output_path is not None and os.path.exists(
            os.path.join(output_path, "base.pkl")
        ):
            with open(os.path.join(output_path, "base.pkl"), "rb") as f:
                base_results = pickle.load(f)
            return base_results, False
        else:
            full_pass = True
            images_dataset = ImagesData(lmdb_engine)
            num_workers = 0 if len(images_dataset) < 5 else 2
            images_loader = torch.utils.data.DataLoader(
                images_dataset, batch_size=1, num_workers=num_workers, shuffle=False
            )
            images_loader = iter(images_loader)
            base_results = {}
            for image_data in tqdm(images_loader):
                image_data = data_to_device(image_data, device=self._device)
                image, image_key = image_data["image"][0], image_data["image_key"][0]
                emica_inputs = self.emica_data_engine(image, image_key)
                if emica_inputs is None:
                    full_pass = False
                    continue
                emica_inputs = torch.utils.data.default_collate([emica_inputs])
                emica_inputs = data_to_device(emica_inputs, device=self._device)
                emica_results = self.emica_encoder(emica_inputs)
                vgg_results, bbox, lmks_2d70 = self.vgghead_encoder(image, image_key)
                if vgg_results is None:
                    full_pass = False
                    continue
                emica_results, vgg_results = self._process_emica_vgg(
                    emica_results, vgg_results, lmks_2d70
                )
                base_results[image_key] = {
                    "emica_results": emica_results,
                    "vgg_results": vgg_results,
                    "bbox": bbox.cpu().numpy() / 512.0,
                }
            if output_path is not None:
                with open(os.path.join(output_path, "base.pkl"), "wb") as f:
                    pickle.dump(base_results, f)
            return base_results, full_pass

    def track_optim(self, base_result, output_path, lmdb_engine=None, share_id=False):
        if output_path is not None and os.path.exists(
            os.path.join(output_path, "optim.pkl")
        ):
            with open(os.path.join(output_path, "optim.pkl"), "rb") as f:
                optim_results = pickle.load(f)
            return optim_results
        else:
            # self.optim_engine.init_model(self.calibration_results, image_size=512)
            base_result = {k: v for k, v in base_result.items() if v is not None}
            for k, v in base_result.items():
                base_result[k]["raw_info"] = self.raw_info[k]
            mini_batchs = build_minibatch(list(base_result.keys()), share_id=share_id)

            if (
                lmdb_engine is not None
                and len(mini_batchs) > 0
                and len(mini_batchs[0]) >= 100
            ):
                batch_frames = (
                    torch.stack([lmdb_engine[key] for key in mini_batchs[0][:100]])
                    .to(self._device)
                    .float()
                )
            else:
                batch_frames = None
            optim_results = {}
            for mini_batch in mini_batchs:
                mini_batch_emica = [base_result[key] for key in mini_batch]
                mini_batch_emica = torch.utils.data.default_collate(mini_batch_emica)
                mini_batch_emica = data_to_device(mini_batch_emica, device=self._device)
                optim_result, visualization = self.optim_engine.lightning_optimize(
                    mini_batch,
                    mini_batch_emica,
                    batch_frames=batch_frames,
                    share_id=share_id,
                    if_crop=self.if_crop,
                )
                batch_frames = None
                if visualization is not None:
                    torchvision.utils.save_image(
                        visualization, os.path.join(output_path, "optim.jpg")
                    )
                optim_results.update(optim_result)

            with open(os.path.join(output_path, "optim.pkl"), "wb") as f:
                pickle.dump(optim_results, f)

            return optim_results

    def track_image(self, inp_images, inp_keys, inp_paths=None, if_matting=True):
        assert (
            type(inp_images) == list
        ), f"Image must be a list, but got {type(inp_images)}."
        assert (
            inp_images[0].dim() == 3
        ), f"Image dim must be 3, but got {inp_images[0].dim()}."
        assert (
            inp_images[0].max() > 1.0
        ), f"Image in [0, 255.0], but got {inp_images[0].max()}."
        assert len(inp_images) == len(
            inp_keys
        ), f"Image and key length must be equal, but got {inp_images.shape[0]} and {len(inp_keys)}."
        croped_images, croped_keys = [], []
        self.raw_info = {}
        self.if_crop = True
        for inp_key, inp_image in tqdm(
            zip(inp_keys, inp_images), total=len(inp_images)
        ):
            croped_image, frame_bbox = self.crop_image(inp_image)
            self.raw_info[inp_key] = {
                "image_size": [inp_image.shape[1], inp_image.shape[2]],
                "frame_bbox": frame_bbox,
            }
            if inp_image is not None:
                if if_matting:
                    croped_image = self.matting_engine.forward(
                        croped_image / 255.0, return_type="matting", background_rgb=0.0
                    )
                    croped_image = croped_image.clamp(0.0, 1.0) * 255.0
                croped_images.append(croped_image)
                croped_keys.append(inp_key)
        if not len(croped_images):
            print("No face detected in all the images, tracking failed.")
            return None
        images_engine = {
            key: image.cpu() for key, image in zip(croped_keys, croped_images)
        }
        base_results = self.track_base(images_engine, None)
        if not len(base_results.keys()):
            print("No face detected in all the images, tracking failed.")
            return None
        # for k, v in base_results.items():
        #     base_results[k]['raw_info'] = raw_info[k]
        sapiens_dir = os.path.dirname(inp_paths[0]) + "_sapiens"
        for k in base_results:
            with open(
                os.path.join(sapiens_dir, k.replace(".png", ".json")), "r"
            ) as file:
                data = json.load(file)
            base_results[k]["sapiens_kpts"] = torch.Tensor(
                data["instance_info"][0]["keypoints"]
            )
        optim_results = self.track_optim(base_results, None, None, share_id=False)
        for key in optim_results:
            optim_results[key]["image"] = images_engine[key].cpu().numpy() / 255.0
        # do visualization
        flame_model = FLAMEModel(
            self.model_path,
            n_shape=300,
            n_exp=100,
            scale=self.calibration_results["verts_scale"],
        )
        mesh_render = RenderMesh(
            512, faces=flame_model.get_faces().cpu().numpy(), device=self._device
        )
        for key in optim_results:
            pred_vertices, _ = flame_model(
                shape_params=torch.tensor(optim_results[key]["shapecode"])[None],
                expression_params=torch.tensor(optim_results[key]["expcode"])[None],
                pose_params=torch.tensor(optim_results[key]["posecode"])[None],
                neck_pose_params=torch.tensor(optim_results[key]["neckcode"])[None],
                eye_pose_params=torch.tensor(optim_results[key]["eyecode"])[None],
            )
            rendered_image, alpha_image = mesh_render(
                pred_vertices.to(self._device),
                focal_length=self.calibration_results["focal_length"],
                transform_matrix=torch.tensor(optim_results[key]["transform_matrix"])[
                    None
                ].to(self._device),
            )
            rendered_image = rendered_image[0].cpu().numpy() / 255.0
            alpha_image = alpha_image[0].expand(3, -1, -1).cpu().numpy()
            vis_image = optim_results[key]["image"].copy()
            # vis_image[alpha_image>0.5] *= 0.5
            # vis_image[alpha_image>0.5] += (rendered_image[alpha_image>0.5] * 0.5)
            vis_image *= 0.1
            vis_image += rendered_image * 0.9
            optim_results[key]["vis_image"] = vis_image
        return optim_results

    def crop_image(self, inp_image):
        ori_height, ori_width = inp_image.shape[1:]
        # if not hasattr(self.emica_data_engine, 'insight_detector'):
        #     self.emica_data_engine._init_models()
        # bbox, _, _ = self.emica_data_engine.insight_detector.get(inp_image)
        _, bbox, _ = self.vgghead_encoder(inp_image, "online_track", only_vgghead=True)
        if bbox is None:
            return None
        bbox = expand_bbox(bbox, scale=1.65).long()
        crop_image = torchvision.transforms.functional.crop(
            inp_image,
            top=bbox[1],
            left=bbox[0],
            height=bbox[3] - bbox[1],
            width=bbox[2] - bbox[0],
        )
        crop_image = torchvision.transforms.functional.resize(
            crop_image, (512, 512), antialias=True
        )
        return crop_image, bbox

    @staticmethod
    def _process_emica_vgg(emica_results, vgg_results, lmks_2d70):
        processed_emica_results = {
            "shapecode": emica_results["shapecode"][0].cpu().numpy(),
            "expcode": emica_results["expcode"][0].cpu().numpy(),
            "globalpose": emica_results["globalpose"][0].cpu().numpy(),
            "jawpose": emica_results["jawpose"][0].cpu().numpy(),
        }
        processed_vgg_results = {
            "shapecode": vgg_results["shapecode"].cpu().numpy(),
            "expcode": vgg_results["expcode"].cpu().numpy(),
            "posecode": vgg_results["posecode"].cpu().numpy(),
            "transform": {
                "rotation_6d": vgg_results["rotation_6d"].cpu().numpy(),
                "translation": vgg_results["translation"].cpu().numpy(),
                "scale": vgg_results["scale"].cpu().numpy(),
            },
            "normalize": vgg_results["normalize"],
            "lmks_2d70": lmks_2d70.cpu().numpy(),
        }
        return processed_emica_results, processed_vgg_results


class ImagesData(torch.utils.data.Dataset):
    def __init__(self, lmdb_engine):
        super().__init__()
        self._lmdb_engine = lmdb_engine
        self._image_keys = list(lmdb_engine.keys())

    def __getitem__(self, index):
        image_key = self._image_keys[index]
        image = self._lmdb_engine[image_key]
        return {"image": image, "image_key": image_key}

    def __len__(
        self,
    ):
        return len(self._image_keys)


def data_to_device(data_dict, device="cuda"):
    assert isinstance(data_dict, dict), "Data must be a dictionary."
    for key in data_dict:
        if isinstance(data_dict[key], torch.Tensor):
            data_dict[key] = data_dict[key].to(device)
        elif isinstance(data_dict[key], np.ndarray):
            data_dict[key] = torch.tensor(data_dict[key], device=device)
        elif isinstance(data_dict[key], dict):
            data_dict[key] = data_to_device(data_dict[key], device=device)
        else:
            continue
    return data_dict


def build_minibatch(all_frames, batch_size=1024, share_id=False):
    if share_id:
        try:
            all_frames = sorted(all_frames, key=lambda x: int(x.split("_")[-1]))
        except:
            all_frames = sorted(all_frames)
        video_names = list(
            set(["_".join(frame_name.split("_")[:-1]) for frame_name in all_frames])
        )
        video_frames = {video_name: [] for video_name in video_names}
        for frame in all_frames:
            video_name = "_".join(frame.split("_")[:-1])
            video_frames[video_name].append(frame)
        all_mini_batch = []
        for video_name in video_names:
            mini_batch = []
            for frame_name in video_frames[video_name]:
                mini_batch.append(frame_name)
                if len(mini_batch) % batch_size == 0:
                    all_mini_batch.append(mini_batch)
                    mini_batch = []
            if len(mini_batch):
                all_mini_batch.append(mini_batch)
    else:
        try:
            all_frames = sorted(all_frames, key=lambda x: int(x.split("_")[-1]))
        except:
            all_frames = sorted(all_frames)
        all_mini_batch, mini_batch = [], []
        for frame_name in all_frames:
            mini_batch.append(frame_name)
            if len(mini_batch) % batch_size == 0:
                all_mini_batch.append(mini_batch)
                mini_batch = []
        if len(mini_batch):
            all_mini_batch.append(mini_batch)
    return all_mini_batch


def expand_bbox(bbox, scale=1.1):
    xmin, ymin, xmax, ymax = bbox.unbind(dim=-1)
    cenx, ceny = (xmin + xmax) / 2, (ymin + ymax) / 2
    # ceny = ceny - (ymax - ymin) * 0.05
    extend_size = torch.sqrt((ymax - ymin) * (xmax - xmin)) * scale
    xmine, xmaxe = cenx - extend_size / 2, cenx + extend_size / 2
    ymine, ymaxe = ceny - extend_size / 2, ceny + extend_size / 2
    expanded_bbox = torch.stack([xmine, ymine, xmaxe, ymaxe], dim=-1)
    return torch.stack([xmine, ymine, xmaxe, ymaxe], dim=-1)
