"""Download the pinned GANs N' Roses inference files and official checkpoint."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
REVISION = "59bf4680c9c700890cdf53ffac86a6f7da8a9153"
SOURCE = f"https://raw.githubusercontent.com/mchong6/GANsNRoses/{REVISION}"
CHECKPOINT_URL = "https://drive.usercontent.google.com/download?id=1IMIVke4WDaGayUa7vk_xVw1uqIHikGtC&export=download&confirm=t"
FILES = ("model.py", "op/__init__.py", "op/fused_act.py", "op/upfirdn2d.py", "LICENSE", "README.md")
CHECKPOINT_SHA256 = "36eb8254155dffb12aed780d1ea479a5bb852e2087dea72e032e2290705a0cfb"


def download(url: str, destination: Path, *, expected_size: int | None = None) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    if not destination.is_file():
        temporary = destination.with_suffix(destination.suffix + ".download")
        print(f"Downloading {destination.name} ...", flush=True)
        with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as output:
            if "text/html" in response.headers.get("Content-Type", ""):
                raise RuntimeError(f"Download returned HTML instead of a file: {url}")
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
                digest.update(chunk)
        if expected_size is not None and temporary.stat().st_size != expected_size:
            raise RuntimeError(f"Incomplete download: {temporary}")
        if expected_size is not None and digest.hexdigest() != CHECKPOINT_SHA256:
            temporary.unlink()
            raise RuntimeError("GANs N' Roses checkpoint SHA256 mismatch; retry the official download.")
        temporary.replace(destination)
        return digest.hexdigest()
    if expected_size is not None and destination.stat().st_size != expected_size:
        raise RuntimeError(f"Unexpected checkpoint size: {destination}")
    with destination.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def setup() -> None:
    repo = ROOT / "third_party/GANsNRoses"
    hashes = {name: download(f"{SOURCE}/{name}", repo / name) for name in FILES}
    hashes["GNR_checkpoint.pt"] = download(
        CHECKPOINT_URL, repo / "GNR_checkpoint.pt", expected_size=316705253,
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
