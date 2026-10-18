"""Run the end-to-end RobotPortraitArt generation pipeline."""

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


from robot_portrait.pipeline import (
    GenerationPipeline,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "运行统一的人像预处理、生成、"
            "SVG 矢量化与回栅格流程。"
        )
    )

    parser.add_argument(
        "--input",
        help="输入肖像照片路径",
    )

    parser.add_argument(
        "--output-dir",
        help="本次运行的输出目录",
    )

    parser.add_argument(
        "--method",
        default="baseline",
        help=(
            "生成方法；可用方法由 "
            "GenerationPipeline 注册表决定"
        ),
    )

    parser.add_argument(
        "--config",
        default=str(
            ROOT
            / "configs"
            / "default.yaml"
        ),
        help="项目 YAML 配置文件",
    )

    parser.add_argument(
        "--size",
        type=int,
        default=512,
        help="预处理输出边长",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子",
    )

    parser.add_argument(
        "--list-methods",
        action="store_true",
        help=(
            "列出当前已注册生成器"
            "及可用状态后退出"
        ),
    )

    args = parser.parse_args()

    pipeline = (
        GenerationPipeline
        .from_yaml(
            args.config,
            preprocess_size=args.size,
            seed=args.seed,
        )
    )

    if args.list_methods:
        print(
            json.dumps(
                pipeline.generator_statuses(),
                ensure_ascii=False,
                indent=2,
            )
        )

        return

    if (
        not args.input
        or not args.output_dir
    ):
        parser.error(
            "未使用 --list-methods 时"
            "必须同时提供 --input "
            "和 --output-dir"
        )

    result = pipeline.run(
        args.input,
        args.output_dir,
        method=args.method,
    )

    manifest = json.loads(
        result.manifest_path
        .read_text(
            encoding="utf-8"
        )
    )

    print(
        "[完成] 统一 Pipeline 执行成功"
    )

    print(
        f"[完成] 生成器："
        f"{result.method}"
    )

    print(
        f"[完成] 预处理："
        f"{result.prepared_path}"
    )

    print(
        f"[完成] 栅格肖像："
        f"{result.raster_path}"
    )

    print(
        f"[完成] SVG："
        f"{result.svg_path}"
    )

    print(
        f"[完成] SVG 回栅格："
        f"{result.vector_render_path}"
    )

    print(
        f"[完成] Manifest："
        f"{result.manifest_path}"
    )

    print(
        "[耗时/ms] "
        + ", ".join(
            (
                f"{name}="
                f"{value:.3f}"
            )
            for name, value
            in manifest[
                "timings_ms"
            ].items()
        )
    )


if __name__ == "__main__":
    main()