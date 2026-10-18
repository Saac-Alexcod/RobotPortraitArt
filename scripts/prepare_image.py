"""CLI for portrait preprocessing."""

from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from robot_portrait.io import write_image
from robot_portrait.preprocessing import (
    preprocess_portrait,
)


def prepare_image(
    input_path: str | Path,
    output_path: str | Path,
    size: int = 512,
) -> dict:
    """兼容已有脚本和测试的包装接口。"""

    input_path = Path(input_path)
    output_path = Path(output_path)

    result = preprocess_portrait(
        input_path,
        size=size,
    )

    write_image(
        output_path,
        result.image,
    )

    return result.to_metadata(
        input_path=input_path,
        output_path=output_path,
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "人脸 EXIF 旋转、检测、"
            "居中裁剪和尺寸统一工具。"
        )
    )

    ap.add_argument(
        "--input",
        required=True,
        help="输入照片路径",
    )

    ap.add_argument(
        "--output",
        required=True,
        help="预处理照片路径",
    )

    ap.add_argument(
        "--size",
        type=int,
        default=512,
        help="输出边长，默认 512",
    )

    ap.add_argument(
        "--metadata",
        default="",
        help=(
            "可选 JSON 记录；"
            "默认写到输出同名 .json"
        ),
    )

    args = ap.parse_args()

    result = prepare_image(
        args.input,
        args.output,
        args.size,
    )

    for warning in result.get(
        "warnings",
        [],
    ):
        print(
            f"警告：{warning}",
            file=sys.stderr,
        )

    metadata = (
        Path(args.metadata)
        if args.metadata
        else Path(args.output)
        .with_suffix(".json")
    )

    metadata.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(result)
    print(
        f"配置记录：{metadata}"
    )


if __name__ == "__main__":
    main()