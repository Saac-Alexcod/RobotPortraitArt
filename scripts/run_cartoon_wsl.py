"""WSL-only GANs N' Roses inference; the web server stays on Windows."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    import torch
    import numpy as np
    from PIL import Image

    if not torch.cuda.is_available():
        raise RuntimeError("GANs N' Roses requires a CUDA-enabled WSL Python. Run -SetupAPD first.")
    sys.path.insert(0, str(args.repo.resolve()))
    from model import Generator

    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = Generator(256, 4, 8, 5, channel_multiplier=1, lr_mlp=.01, n_res=1)
    checkpoint = torch.load(args.repo / "GNR_checkpoint.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["G_A2B_ema"], strict=True)
    del checkpoint
    model = model.eval().to("cuda")
    if not args.check:
        if args.input is None or args.output is None:
            parser.error("--input and --output are required for inference")
        with Image.open(args.input) as source:
            rgb = source.convert("RGB")
            if rgb.size != (256, 256):
                raise ValueError("Expected a 256x256 prepared portrait")
            array = np.asarray(rgb).copy()
        tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).float().to("cuda") / 127.5 - 1
        with torch.inference_mode():
            style = torch.randn(1, 8, device="cuda")
            content, _ = model.encode(tensor)
            output = model.decode(content, style).clamp(-1, 1)
        array = ((output[0].permute(1, 2, 0).cpu().numpy() + 1) * 127.5).round().astype(np.uint8)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(array).save(args.output)
    print(json.dumps({"ready": True, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0), "seed": args.seed}))


if __name__ == "__main__":
    main()
