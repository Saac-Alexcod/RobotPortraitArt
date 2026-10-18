"""Compare an automatically generated mask with a reference mask."""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys

import cv2
import numpy as np


ROOT = Path(
    __file__
).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT / "src"),
)


from robot_portrait.apdrawinggan_preprocess import (
    compare_binary_masks,
)
from robot_portrait.io import (
    read_image,
)


def _read_gray(
    path: str | Path,
) -> np.ndarray:
    image = read_image(
        path,
        cv2.IMREAD_GRAYSCALE,
    )

    if (
        image is None
        or image.size == 0
    ):
        raise ValueError(
            f"无法读取 mask：{path}"
        )

    return image


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "比较自动生成的 APDrawingGAN mask "
            "与参考 mask。"
        )
    )

    parser.add_argument(
        "--predicted",
        required=True,
        help="自动生成 mask",
    )

    parser.add_argument(
        "--reference",
        required=True,
        help="参考/官方 mask",
    )

    args = parser.parse_args()

    predicted = _read_gray(
        args.predicted
    )

    reference = _read_gray(
        args.reference
    )

    metrics = (
        compare_binary_masks(
            predicted,
            reference,
        )
    )

    payload = {
        "predicted": str(
            Path(
                args.predicted
            )
        ),
        "reference": str(
            Path(
                args.reference
            )
        ),
        **metrics,
    }

    print(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
