import sys

sys.path.append("./")
import os

from engine.SegmentAPI.box_prior import SAM2Seg


def write_txt(save_name, my_list):
    with open(save_name, "w", encoding="utf-8") as file:
        for item in my_list[:-1]:
            file.write(item + "\n")
        if len(my_list) > 0:
            file.write(my_list[-1])


def init_box_model(model_path):
    sam2_weight = os.path.join(model_path, "sam2/sam2.1_hiera_large.pt")
    birefnet_weight = os.path.join(model_path, "BiRefNet-general-epoch_244.pth")
    assert os.path.exists(sam2_weight) and os.path.exists(birefnet_weight)
    model = SAM2Seg(
        wo_supres=True, sam_weight=sam2_weight, prior_weight=birefnet_weight
    )
    return model


def predict_box(model, output_path):
    img = os.path.join(output_path, "imgs_png/00001.png")
    out = model(img_path=img, bbox=None)
    str_result = ",".join(map(str, out.xywh))

    save_txt_path = os.path.join(output_path, "bbox")
    os.makedirs(save_txt_path, exist_ok=True)
    save_txt = os.path.join(save_txt_path, "first_frame.txt")
    write_txt(save_txt, [str_result])
