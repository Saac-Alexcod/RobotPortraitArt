"""Download the pinned GANs N' Roses inference files and official checkpoint."""
from __future__ import annotations

import json
import argparse
import os
from pathlib import Path
import shutil
from urllib.parse import urlsplit
from model_download import download, sha256

ROOT = Path(__file__).resolve().parents[1]
REVISION = "59bf4680c9c700890cdf53ffac86a6f7da8a9153"
SOURCE = f"https://raw.githubusercontent.com/mchong6/GANsNRoses/{REVISION}"
CHECKPOINT_URL = "https://drive.usercontent.google.com/download?id=1IMIVke4WDaGayUa7vk_xVw1uqIHikGtC&export=download&confirm=t"
FILES = ("model.py", "op/__init__.py", "op/fused_act.py", "op/upfirdn2d.py", "LICENSE", "README.md")
CHECKPOINT_SHA256 = "36eb8254155dffb12aed780d1ea479a5bb852e2087dea72e032e2290705a0cfb"
CHECKPOINT_SIZE = 316705253


def setup(*, checkpoint_url=None, checkpoint_file=None) -> None:
    explicit_url = checkpoint_url is not None
    if checkpoint_url and checkpoint_file:
        raise ValueError('Choose either checkpoint_url or checkpoint_file.')
    checkpoint_file = checkpoint_file or (None if explicit_url else os.environ.get('ROBOT_PORTRAIT_GNR_FILE'))
    checkpoint_url = checkpoint_url or os.environ.get('ROBOT_PORTRAIT_GNR_URL') or CHECKPOINT_URL
    if not checkpoint_file and (urlsplit(checkpoint_url).scheme != 'https' or not urlsplit(checkpoint_url).hostname):
        raise ValueError('GNR checkpoint URL must be an HTTPS direct download URL.')
    repo = ROOT / "third_party/GANsNRoses"
    target = repo / 'GNR_checkpoint.pt'
    if checkpoint_file:
        source = Path(checkpoint_file).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f'Local checkpoint not found: {source}')
        if source.stat().st_size != CHECKPOINT_SIZE or sha256(source) != CHECKPOINT_SHA256:
            raise ValueError('Local GANs N\' Roses checkpoint size/SHA256 mismatch.')
        repo.mkdir(parents=True, exist_ok=True)
        if source != target.resolve():
            temporary = target.with_name(target.name + '.download')
            try:
                shutil.copyfile(source, temporary)
                if sha256(temporary) != CHECKPOINT_SHA256:
                    raise ValueError('Copied checkpoint SHA256 mismatch.')
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
    hashes = {name: download(f"{SOURCE}/{name}", repo / name) for name in FILES}
    hashes["GNR_checkpoint.pt"] = download(
        checkpoint_url, target, expected_size=CHECKPOINT_SIZE,
        expected_sha256=CHECKPOINT_SHA256,
    )
    if hashes["GNR_checkpoint.pt"] != CHECKPOINT_SHA256:
        raise RuntimeError("GANs N' Roses checkpoint SHA256 mismatch; please download the official checkpoint again.")
    (repo / "provenance.json").write_text(json.dumps({
        "source": SOURCE, "revision": REVISION,
        "checkpoint_url": checkpoint_url if not checkpoint_file else None,
        "checkpoint_file": str(source) if checkpoint_file else None, "sha256": hashes,
    }, indent=2), encoding="utf-8")
    print(f"GANs N' Roses installed at {repo}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--checkpoint-url', help='HTTPS direct URL for the identical official checkpoint')
    group.add_argument('--checkpoint-file', type=Path, help='Import an already downloaded official checkpoint')
    args = parser.parse_args()
    setup(checkpoint_url=args.checkpoint_url, checkpoint_file=args.checkpoint_file)
