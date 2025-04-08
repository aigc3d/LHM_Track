import os
import sys

sys.path.append("./")


def run_sapiens(model_path, output_dir):
    model_path = os.path.join(
        model_path,
        "sapiens/poses/sapiens_1b_coco_wholebody_best_coco_wholebody_AP_727_torchscript.pt2",
    )
    img_path = os.path.join(output_dir, "imgs_png")

    output_path = os.path.join(output_dir, "sapiens_pose")

    cmd = f"python ./engine/sapiens_api/core/vis_pose.py \
            {model_path} \
            --num_keypoints 133 \
            --batch-size 1 \
            --input {img_path} \
            --output-root={output_path} \
            --radius 6 \
            --kpt-thr 0.3"

    os.system(cmd)
