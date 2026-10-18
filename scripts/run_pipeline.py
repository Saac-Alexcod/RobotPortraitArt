"""Run the end-to-end RobotPortraitArt generation pipeline."""

from pathlib import Path
import argparse
import json
import os
import sys


ROOT = Path(
    __file__
).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT / "src"),
)


from robot_portrait.generators import (
    APDrawingGANGenerator,
    APDrawingGANGeneratorConfig,
    BaselineGenerator,
)
from robot_portrait.pipeline import (
    GenerationPipeline,
    PipelineConfig,
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
        "--landmark",
        help=(
            "APDrawingGAN 五点 landmark 文件；"
            "坐标对应原始输入照片"
        ),
    )

    parser.add_argument(
        "--mask",
        help=(
            "APDrawingGAN 背景 mask PNG；"
            "尺寸对应原始输入照片"
        ),
    )

    parser.add_argument(
        "--apd-repo",
        default=str(
            ROOT
            / "third_party"
            / "APDrawingGAN"
        ),
        help="APDrawingGAN 源码目录",
    )

    parser.add_argument(
        "--apd-distro",
        default="Ubuntu-22.04",
        help="运行 APDrawingGAN 的 WSL 发行版",
    )

    parser.add_argument(
        "--apd-python",
        default=os.environ.get(
            "ROBOT_PORTRAIT_APD_PYTHON",
            "python3",
        ),
        help=(
            "WSL 内 APDrawingGAN Python；"
            "也可通过 ROBOT_PORTRAIT_APD_PYTHON 设置"
        ),
    )

    parser.add_argument(
        "--apd-epoch",
        default="300",
        help="APDrawingGAN checkpoint epoch",
    )

    parser.add_argument(
        "--apd-gpu-ids",
        default="0",
        help="传给 APDrawingGAN --gpu_ids 的值",
    )

    parser.add_argument(
        "--apd-timeout",
        type=float,
        default=180.0,
        help="APDrawingGAN 单张推理超时秒数",
    )

    parser.add_argument(
        "--apd-keep-work-dir",
        action="store_true",
        help="成功后保留 .apdrawinggan_work 调试目录",
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

    config = PipelineConfig.from_yaml(
        args.config,
        preprocess_size=args.size,
        seed=args.seed,
    )

    generators = {
        BaselineGenerator.name:
            BaselineGenerator(
                config.baseline
            ),

        APDrawingGANGenerator.name:
            APDrawingGANGenerator(
                APDrawingGANGeneratorConfig(
                    repo=Path(
                        args.apd_repo
                    ),
                    distro=args.apd_distro,
                    python_command=(
                        args.apd_python
                    ),
                    epoch=args.apd_epoch,
                    gpu_ids=args.apd_gpu_ids,
                    timeout_seconds=(
                        args.apd_timeout
                    ),
                    keep_work_dir=(
                        args.apd_keep_work_dir
                    ),
                )
            ),
    }

    pipeline = GenerationPipeline(
        config,
        generators=generators,
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

    if (
        args.method
        == "apdrawinggan"
        and args.size != 512
    ):
        parser.error(
            "--method apdrawinggan 当前要求 "
            "--size 512"
        )

    if (
        args.method
        == "apdrawinggan"
        and (
            not args.landmark
            or not args.mask
        )
    ):
        parser.error(
            "--method apdrawinggan 目前需要同时提供 "
            "--landmark 和 --mask"
        )

    auxiliary_inputs: dict[
        str,
        Path,
    ] = {}

    if args.landmark:
        auxiliary_inputs[
            "landmark"
        ] = Path(
            args.landmark
        )

    if args.mask:
        auxiliary_inputs[
            "mask"
        ] = Path(
            args.mask
        )

    result = pipeline.run(
        args.input,
        args.output_dir,
        method=args.method,
        auxiliary_inputs=(
            auxiliary_inputs
        ),
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
