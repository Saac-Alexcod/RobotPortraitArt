from pathlib import Path
import argparse, json, os, subprocess, sys


def main():
    ap = argparse.ArgumentParser(description="用于调用 APDrawingGAN 官方 test.py 的轻量适配器。")
    ap.add_argument("--repo", required=True, help="已克隆的 yiranran/APDrawingGAN 工程路径")
    ap.add_argument("--input_dir", required=True, help="已经完成对齐和预处理的 test_single 数据目录")
    ap.add_argument("--experiment", default="formal_author")
    ap.add_argument("--epoch", default="300")
    ap.add_argument("--python", default=sys.executable,
                    help="运行 APDrawingGAN 的 Python 解释器；可能需要独立的旧版环境。")
    ap.add_argument("--seed", type=int, default=42, help="记录用随机种子；官方脚本是否使用需以其实现为准")
    ap.add_argument("--log-json", default="", help="保存调用配置和命令的 JSON 路径")
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    test_py = repo / "test.py"
    if not test_py.exists():
        raise FileNotFoundError(f"缺少 {test_py}。请先克隆官方仓库，并准备 legacy 环境。")
    input_dir = Path(args.input_dir).resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"缺少预处理数据目录：{input_dir}")

    cmd = [
        args.python, "test.py",
        "--dataroot", str(input_dir),
        "--name", args.experiment,
        "--model", "test",
        "--dataset_mode", "single",
        "--norm", "batch",
        "--use_local",
        "--which_epoch", str(args.epoch),
    ]
    print("正在运行：", " ".join(cmd))
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = str(args.seed)
    subprocess.run(cmd, cwd=repo, check=True, env=env)
    log_path = Path(args.log_json) if args.log_json else Path("outputs/apdrawinggan_adapter.json")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps({"seed": args.seed, "repo": str(repo), "input_dir": str(input_dir), "command": cmd, "pythonhashseed": str(args.seed)}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[完成] 调用记录：{log_path}")


if __name__ == "__main__":
    main()
