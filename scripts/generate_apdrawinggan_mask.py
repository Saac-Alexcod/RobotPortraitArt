"""Generate an APDrawingGAN-compatible portrait mask with PPHumanSeg."""

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
    HumanSegConfig,
    draw_human_mask_preview,
    generate_pphumanseg_mask,
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
    / "human_segmentation_pphumanseg_2023mar.onnx"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "使用 OpenCV Zoo PPHumanSeg 自动生成 "
            "APDrawingGAN 所需的人像前景 mask。"
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help=(
            "输入图像路径；建议使用已完成 "
            "APDrawingGAN 512x512 对齐的图像"
        ),
    )

    parser.add_argument(
        "--output",
        required=True,
        help="输出 0/255 二值 mask PNG",
    )

    parser.add_argument(
        "--model",
        default=str(
            DEFAULT_MODEL
        ),
        help="PPHumanSeg ONNX 模型路径",
    )

    parser.add_argument(
        "--preview",
        help=(
            "可选：保存前景保留、背景淡化的预览图"
        ),
    )

    args = parser.parse_args()

    image = load_portrait_image(
        args.input
    )

    result = (
        generate_pphumanseg_mask(
            image,
            HumanSegConfig(
                model_path=Path(
                    args.model
                )
            ),
        )
    )

    output_path = Path(
        args.output
    )

    write_image(
        output_path,
        result.mask,
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
        "output_mask"
    ] = str(
        output_path
    )

    payload[
        "model_path"
    ] = str(
        Path(
            args.model
        )
    )

    if args.preview:
        preview = (
            draw_human_mask_preview(
                image,
                result,
            )
        )

        write_image(
            args.preview,
            preview,
        )

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
