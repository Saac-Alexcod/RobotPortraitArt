"""Download the pinned GANs N' Roses inference files and official checkpoint."""
from __future__ import annotations

import json
from pathlib import Path
from model_download import download

ROOT = Path(__file__).resolve().parents[1]
REVISION = "59bf4680c9c700890cdf53ffac86a6f7da8a9153"
SOURCE = f"https://raw.githubusercontent.com/mchong6/GANsNRoses/{REVISION}"
CHECKPOINT_URL = "https://drive.usercontent.google.com/download?id=1IMIVke4WDaGayUa7vk_xVw1uqIHikGtC&export=download&confirm=t"
FILES = ("model.py", "op/__init__.py", "op/fused_act.py", "op/upfirdn2d.py", "LICENSE", "README.md")
CHECKPOINT_SHA256 = "36eb8254155dffb12aed780d1ea479a5bb852e2087dea72e032e2290705a0cfb"


def setup() -> None:
    repo = ROOT / "third_party/GANsNRoses"
    hashes = {name: download(f"{SOURCE}/{name}", repo / name) for name in FILES}
    hashes["GNR_checkpoint.pt"] = download(
        CHECKPOINT_URL, repo / "GNR_checkpoint.pt", expected_size=316705253,
        expected_sha256=CHECKPOINT_SHA256,
    )
    if hashes["GNR_checkpoint.pt"] != CHECKPOINT_SHA256:
        raise RuntimeError("GANs N' Roses checkpoint SHA256 mismatch; please download the official checkpoint again.")
    (repo / "provenance.json").write_text(json.dumps({
        "source": SOURCE, "revision": REVISION,
        "checkpoint_url": CHECKPOINT_URL, "sha256": hashes,
    }, indent=2), encoding="utf-8")
    print(f"GANs N' Roses installed at {repo}", flush=True)


if __name__ == "__main__":
    setup()
