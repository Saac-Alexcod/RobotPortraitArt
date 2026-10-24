"""Align a face to APDrawingGAN's official 512x512 geometry."""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys


ROOT = Path(
    __file__
).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT / "src"),
)


from robot_portrait.apdrawinggan_preprocess import (
    align_apdrawinggan_face,
    read_apdrawinggan_landmarks,
    write_apdrawinggan_landmarks,
)
from robot_portrait.io import (
    write_image,
)
from robot_portrait.preprocessing import (
    load_portrait_image,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "使用 APDrawingGAN 官方五点模板"
            "将人脸对齐到 512x512。"
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="输入原始照片路径",
    )

    parser.add_argument(
        "--landmark",
        required=True,
        help="输入 5 点 landmark txt 路径",
    )

    parser.add_argument(
        "--output-image",
        required=True,
        help="输出对齐图像路径",
    )

    parser.add_argument(
        "--output-landmark",
        required=True,
        help="输出变换后 landmark txt 路径",
    )

    parser.add_argument(
        "--size",
        type=int,
        default=512,
        help=(
            "输出正方形边长，默认 512；"
            "APDrawingGAN 官方模型使用 512"
        ),
    )

    parser.add_argument(
        "--no-reflection",
        action="store_true",
        help=(
            "禁用反射候选；默认复现 "
            "MATLAB cp2tform(..., 'similarity')"
            " 的可选反射行为"
        ),
    )

    args = parser.parse_args()

    image = load_portrait_image(
        args.input
    )

    landmarks = (
        read_apdrawinggan_landmarks(
            args.landmark
        )
    )

    result = (
        align_apdrawinggan_face(
            image,
            landmarks,
            output_size=args.size,
            allow_reflection=(
                not args.no_reflection
            ),
        )
    )

    image_path = Path(
        args.output_image
    )

    write_image(
        image_path,
        result.image,
    )

    landmark_path = (
        write_apdrawinggan_landmarks(
            args.output_landmark,
            result.points,
        )
    )

    payload = (
        result.to_metadata()
    )

    payload[
        "input_image"
    ] = str(
        Path(
            args.input
        )
    )

    payload[
        "input_landmark"
    ] = str(
        Path(
            args.landmark
        )
    )

    payload[
        "output_image"
    ] = str(
        image_path
    )

    payload[
        "output_landmark"
    ] = str(
        landmark_path
    )

    print(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
