"""Inspect APDrawingGAN source, checkpoints and WSL GPU runtime."""

from pathlib import Path
import argparse
import json
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(
    0,
    str(ROOT / "src"),
)


from robot_portrait.generators.apdrawinggan import (
    APDrawingGANPreflight,
    APDrawingGANPreflightConfig,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "检查 APDrawingGAN 源码、预训练权重、WSL、"
            "Python、PyTorch 与 CUDA 是否满足推理接入条件。"
        )
    )

    parser.add_argument(
        "--repo",
        default=str(
            ROOT
            / "third_party"
            / "APDrawingGAN"
        ),
        help="APDrawingGAN 源码目录",
    )

    parser.add_argument(
        "--distro",
        default="Ubuntu-22.04",
        help="WSL 发行版名称",
    )

    parser.add_argument(
        "--python",
        default="python3",
        help="WSL 中的 Python 命令",
    )

    parser.add_argument(
        "--experiment",
        default="formal_author",
        help="checkpoint 实验目录名",
    )

    parser.add_argument(
        "--epoch",
        default="300",
        help="checkpoint epoch",
    )

    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="CUDA 不可用时仍允许 preflight 通过",
    )

    parser.add_argument(
        "--json",
        dest="json_path",
        default="",
        help="可选：写出完整 JSON 报告",
    )

    args = parser.parse_args()

    config = (
        APDrawingGANPreflightConfig(
            repo=Path(
                args.repo
            ).resolve(),

            distro=args.distro,

            python_command=(
                args.python
            ),

            experiment=(
                args.experiment
            ),

            epoch=str(
                args.epoch
            ),

            require_cuda=(
                not args.allow_cpu
            ),
        )
    )

    report = (
        APDrawingGANPreflight(
            config
        ).run()
    )

    metadata = (
        report.to_metadata()
    )

    print(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
        )
    )

    if args.json_path:
        output = Path(
            args.json_path
        )

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        output.write_text(
            json.dumps(
                metadata,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        print(
            f"[完成] Preflight 报告："
            f"{output}"
        )

    if not report.ready:
        failed = (
            report
            .failed_required_checks()
        )

        print(
            "[未就绪] "
            + "; ".join(
                check.name
                for check in failed
            ),
            file=sys.stderr,
        )

        raise SystemExit(2)

    print(
        "[就绪] APDrawingGAN 环境 "
        "preflight 已通过。"
    )


if __name__ == "__main__":
    main()