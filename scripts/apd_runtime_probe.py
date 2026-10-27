"""Check the requirements of the APDrawingGAN *inference* path in WSL.

Training-only Visdom is intentionally not required: third_party/APDrawingGAN/test.py
sets ``display_id = -1`` and does not connect to Visdom. The actual inference
smoke test remains the final acceptance gate.
"""
from __future__ import annotations

import importlib
import sys

REQUIRED_INFERENCE_MODULES = (
    "torch", "torchvision", "dominate", "numpy", "PIL",
)


def check_runtime() -> int:
    modules = {}
    for name in REQUIRED_INFERENCE_MODULES:
        try:
            modules[name] = importlib.import_module(name)
        except (ImportError, OSError, RuntimeError) as exc:
            print(f"[RPA-E250] Required inference module {name!r} is unavailable: {exc}", file=sys.stderr)
            return 21

    torch = modules["torch"]
    try:
        if not torch.cuda.is_available():
            print("[RPA-E250] PyTorch imported but CUDA is unavailable in this WSL interpreter.", file=sys.stderr)
            return 22
        # CUDA discovery alone is not proof that the driver can execute a kernel.
        sample = torch.ones((1,), device="cuda")
        value = float((sample + 1).item())
        if value != 2.0:
            raise RuntimeError(f"unexpected CUDA result: {value}")
        print(f"[RPA] Inference runtime: Python {sys.version.split()[0]}, "
              f"PyTorch {torch.__version__}, CUDA {torch.cuda.get_device_name(0)}")
    except (RuntimeError, OSError, AssertionError) as exc:
        print(f"[RPA-E250] CUDA execution failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 23

    # Visdom is imported only for training/interactive display when display_id > 0.
    print("RPA_RUNTIME_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(check_runtime())
