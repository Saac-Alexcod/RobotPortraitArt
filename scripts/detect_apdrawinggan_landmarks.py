"""Detect APDrawingGAN five-point landmarks with OpenCV YuNet."""

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
    YuNetConfig,
    detect_yunet_five_points,
    draw_five_point_preview,
    write_apdrawinggan_landmarks,
)
from robot_portrait.io import (
    write_image,
)
from robot_portrait.preprocessing import (
    load_portrait_image,
)


DEFAULT_MODEL = (
    ROOT
    / "third_party"
    / "models"
    / "face_detection_yunet_2023mar.onnx"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "使用 OpenCV YuNet 自动检测 "
            "APDrawingGAN 所需的 5 点人脸 landmark。"
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="输入照片路径",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="输出 APDrawingGAN landmark txt 路径",
    )

    parser.add_argument(
        "--model",
        default=str(
            DEFAULT_MODEL
        ),
        help="YuNet ONNX 模型路径",
    )

    parser.add_argument(
        "--preview",
        help=(
            "可选：保存带人脸框与 5 点标记的预览图"
        ),
    )

    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.9,
        help="YuNet 置信度阈值，默认 0.9",
    )

    parser.add_argument(
        "--max-input-side",
        type=int,
        default=640,
        help=(
            "检测前最大边缩放限制，默认 640；"
            "landmark 会映射回原图坐标"
        ),
    )

    args = parser.parse_args()

    image = load_portrait_image(
        args.input
    )

    detection = (
        detect_yunet_five_points(
            image,
            YuNetConfig(
                model_path=Path(
                    args.model
                ),
                score_threshold=(
                    args.score_threshold
                ),
                max_input_side=(
                    args.max_input_side
                ),
            ),
        )
    )

    output_path = (
        write_apdrawinggan_landmarks(
            args.output,
            detection,
        )
    )

    if args.preview:
        preview = (
            draw_five_point_preview(
                image,
                detection,
            )
        )

        write_image(
            args.preview,
            preview,
        )

    payload = (
        detection.to_metadata()
    )

    payload[
        "landmark_file"
    ] = str(
        output_path
    )

    if args.preview:
        payload[
            "preview_file"
        ] = str(
            Path(
                args.preview
            )
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
