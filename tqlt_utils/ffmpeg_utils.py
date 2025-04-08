"""ffmpeg toolkit to process data
"""

import math
import os
import pdb
import shutil

import ffmpeg
import fire
IMG_TYPE_LIST = {'.jpg','.bmp','.png','.jpeg','.rgb','.tif'}
VIDEO_TYPE_LIST = {'.avi','.mp4','.gif','.AVI','.MP4','.GIF'}


def get_framerate(video_path):
    probe = ffmpeg.probe(video_path)
    video_stream = next(
        (stream for stream in probe["streams"] if stream["codec_type"] == "video"), None
    )
    if video_stream is None:
        raise ValueError("No video stream found")
    fps = math.ceil(eval(video_stream["r_frame_rate"]))

    return fps


def decodeffmpeg(inputs, frame_rate, output):
    if os.path.exists(output):
        shutil.rmtree(output)
    os.makedirs(output, exist_ok=False)

    print("extracting video to imgs.....")
    cmd = f"ffmpeg -i {inputs} -vf fps={frame_rate} {output}/%5d.png > /dev/null 2>&1"

    os.system(cmd)
    print("extracting done!")


def encodeffmpeg(inputs, frame_rate, output, format="png", crf=10):
    """output: need video_name"""
    assert (
        os.path.splitext(output)[-1] in VIDEO_TYPE_LIST
    ), "output is the format of video, e.g., mp4"
    assert os.path.isdir(inputs), "input dir is NOT file format"

    inputs = inputs[:-1] if inputs[-1] == "/" else inputs

    output = os.path.abspath(output)

    cmd = (
        f"ffmpeg -r {frame_rate} -pattern_type glob -i '{inputs}/*.{format}' "
        + f'-vcodec libx264 -crf {crf} -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" '
        + f"-pix_fmt yuv420p {output} > /dev/null 2>&1"
    )

    print(cmd)

    output_dir = os.path.dirname(output)
    if os.path.exists(output):
        os.remove(output)
    os.makedirs(output_dir, exist_ok=True)

    print("encoding imgs to video.....")
    os.system(cmd)
    print("video done!")


def vstack(input1, input2, output_name):
    """output: need video_name"""

    os.system(f"rm -f {output_name}")

    cmd = (
        f"ffmpeg -i {input1} "
        + f"-i {input2} "
        + f"-c:v libx264 -crf 10 -filter_complex vstack {output_name} > /dev/null 2>&1"
    )

    print(cmd)

    os.system(cmd)


def hstack(input1, input2, output_name):
    """output: need video_name"""

    os.system(f"rm -f {output_name}")

    cmd = (
        f"ffmpeg -i {input1} "
        + f"-i {input2} "
        + f" -c:v libx264 -crf 10 -filter_complex hstack {output_name} > /dev/null 2>&1"
    )

    print(cmd)

    os.system(cmd)


def clean_tmp(*clean_list):
    for clean in clean_list:
        clean = os.path.abspath(clean)
        cmd = f"rm -f {clean}"
        os.system(cmd)


def merge_video(video_list, save_name, mode="vstack", hash_name="000000"):
    video_list = [os.path.abspath(video_path) for video_path in video_list]

    if len(video_list) == 1:
        return
    os.system(f"rm -f {save_name}")

    tmp_files = []

    tmp_path = f"/tmp/{hash_name}"
    os.makedirs(tmp_path, exist_ok=True)

    if mode == "vstack" or mode == "hstack":
        call_func = vstack if mode == "vstack" else hstack
        for i in range(len(video_list) - 1):
            if i == 0:
                pre = video_list[i]
            else:
                pre = os.path.join(tmp_path, f"{i-1:03d}.mp4")
            next = video_list[i + 1]

            if i + 1 == len(video_list) - 1:
                call_func(pre, next, save_name)
            else:
                call_func(pre, next, os.path.join(tmp_path, f"{i:03d}.mp4"))
                tmp_files.append(os.path.join(tmp_path, f"{i:03d}.mp4"))
    elif mode == "2x2":
        assert len(video_list) == 4
        file1, file2, file3, file4 = video_list
        tmp1 = os.path.join(tmp_path, "tmp1.mp4")
        tmp2 = os.path.join(tmp_path, "tmp2.mp4")
        vstack(file1, file2, tmp1)
        vstack(file3, file4, tmp2)
        hstack(tmp1, tmp2, save_name)
        tmp_files.extend([tmp1, tmp2])
    else:
        raise NotImplementedError

    clean_tmp(*tmp_files)


def concat_video(output, *video_list):
    cmd = "ffmpeg"
    complex_str = ""
    n = 0

    for video in video_list:
        cmd += f" -i {video}"
        complex_str += f"[{n}:v]"
        n += 1

    complex_str += f'concat=n={n}:v=1:a=0" -movflags +faststart {output}'
    complex_str = '"' + complex_str

    cmd = cmd + f" -c:v libx264 -crf 10 -filter_complex {complex_str}"

    if os.path.exists(output):
        os.remove(output)

    os.system(cmd)

    # ffmpeg -i ./gradio_examples/driven_folder/apple0/beauty-2.mp4 -i ./gradio_examples/driven_folder/apple1/beauty-2.mp4 -i ./gradio_examples/driven_folder/apple2/beauty-2.mp4
    # -filter_complex "[0:v][1:v][2:v]concat=n=3:v=1:a=0" -movflags +faststart output.mp4


def img2video(inputs, frame_rate, output, format="png", crf=10):
    encodeffmpeg(inputs, frame_rate, output, format=format, crf=crf)


def video2img(inputs, frame_rate, output):
    decodeffmpeg(inputs, frame_rate, output)


def concatvideo(output, *video_sequence):
    concat_video(output, *video_sequence)


def main():
    # encodeffmpeg
    # decodeffmpeg
    fire.Fire()


if __name__ == "__main__":
    main()
