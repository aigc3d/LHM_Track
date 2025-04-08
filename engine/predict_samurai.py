import sys
import warnings

sys.path.append("./")
import os

SAMURAI = "./engine/samurai/"


def run_samurai(model_path, output_path):
    # warnings.simplefilter("ignore", category=UserWarning, lineno=0, append=False)
    cmd = "python ./scripts/demo.py --video_path {} \
        --txt_path {} \
            --model_path {} --output {} --video_output_path {} --save_img"
    model_path = os.path.join(model_path, "sam2/sam2.1_hiera_large.pt")

    model_path = os.path.abspath(model_path)
    output_dir = os.path.abspath(output_path)  # abs path is required

    cur_path = os.path.abspath("./")
    os.chdir(SAMURAI)

    img_path = os.path.join(output_dir, "imgs_png")
    txt_path = os.path.join(output_dir, "bbox/first_frame.txt")

    assert os.path.exists(img_path) and os.path.exists(txt_path), print(img_path)

    output = os.path.join(output_dir, "samurai_seg")
    save_mp4 = os.path.join(output_dir, "samurai_visualize.mp4")
    try:
        cur_cmd = cmd.format(
            img_path,
            txt_path,
            model_path,
            output,
            save_mp4,
        )
        os.system(cur_cmd)
    finally:
        os.chdir(cur_path)
