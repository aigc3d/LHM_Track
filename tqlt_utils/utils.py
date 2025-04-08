import json
import os
import pdb
import pickle
import time

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image


def write_image(
    path: str,
    img,
    order="RGB",
    need_mapping=False,  # depth need to mapping to color space
):
    """write an image to various formats.

    Args:
        path (str): path to write the image file.
        img (Union[torch.Tensor, np.ndarray, PIL.Image.Image]): image to write.
        order (str, optional): channel order. Defaults to "RGB".
    """

    if isinstance(img, Image.Image):
        img.save(path)
        return

    if torch.is_tensor(img):
        img = img.detach().cpu().numpy()

    if img.dtype == np.float32 or img.dtype == np.float64:
        img = (img * 255).astype(np.uint8)

    # cvtColor
    if len(img.shape) == 3:
        if order == "RGB":
            if img.shape[-1] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
            elif img.shape[-1] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    dir_path = os.path.dirname(path)
    if dir_path != "" and not os.path.exists(dir_path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    if need_mapping:
        plt.imsave(path, img[..., -1], cmap="viridis")
    else:
        cv2.imwrite(path, img)


def read_json(path):
    """load a json file.

    Args:
        path (str): path to json file.

    Returns:
        dict: json content.
    """
    with open(path, "r") as f:
        return json.load(f)


def write_json(path, x):
    """write a json file.

    Args:
        path (str): path to write json file.
        x (dict): dict to write.
    """
    with open(path, "w") as f:
        json.dump(x, f, indent=2)


def read_pickle(path):
    """read a pickle file.

    Args:
        path (str): path to pickle file.

    Returns:
        Any: pickle content.
    """
    with open(path, "rb") as f:
        return pickle.load(f)


def write_pickle(path, x):
    """write a pickle file.

    Args:
        path (str): path to write pickle file.
        x (Any): content to write.
    """
    with open(path, "wb") as f:
        pickle.dump(x, f)


def basename(f):
    return os.path.splitext(os.path.basename(f))[0]


def is_format(f: str, format):
    """if a file's extension is in a set of format

    Args:
        f (str): file name.
        format (Sequence[str]): set of extensions (both '.jpg' or 'jpg' is ok).

    Returns:
        bool: if the file's extension is in the set.
    """
    ext = os.path.splitext(f)[1].lower()  # include the dot
    return ext in format or ext[1:] in format


def is_img(input_list):
    return list(filter(lambda x: is_format(x, [".jpg", ".jpeg", ".png"]), input_list))


def is_dir(input_list):
    return list(filter(lambda x: os.path.isdir(x), input_list))


def is_file(input_list):
    return list(filter(lambda x: os.path.isfile(x), input_list))


def listdir(input_dir):
    return [os.path.join(input_dir, name) for name in os.listdir(input_dir)]


def avaliable_device():
    import torch

    if torch.cuda.is_available():
        current_device_id = torch.cuda.current_device()
        device = f"cuda:{current_device_id}"
    else:
        device = "cpu"

    return device


def next_files(path, format=None):

    files = os.listdir(path)
    files = [os.path.join(path, file) for file in files]
    if format is not None:
        files = list(filter(lambda x: is_format(x, format), files))

    files = is_file(files)
    return sorted(files)


def next_folders(path):
    files = os.listdir(path)
    files = [os.path.join(path, file) for file in files]
    return sorted(is_dir(files))


def read_csv(path):

    df = pd.read_csv(path)

    return df.to_dict()


def download_from_url(url, output):
    cmd = f'wget "{url}" -O {output}'
    os.system(cmd)
