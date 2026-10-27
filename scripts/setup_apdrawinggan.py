"""Install official Model1 and OpenCV preprocessing weights without GPU dependencies."""
import json
from pathlib import Path
import shutil
import zipfile

from model_download import download, sha256
from ensure_apdrawinggan_source import ensure_source

ROOT = Path(__file__).resolve().parents[1]
MODEL_URL = 'https://cg.cs.tsinghua.edu.cn/people/~Yongjin/APDrawingGAN-Models1.zip'
# Reference hashes from the model files already used and tested by this project;
# these are project pins, not publisher-provided signatures.
ARCHIVE_SHA256 = '7b1d41cc8bfdf28f4c9c67fb5f2bb90fd08b0a4582a41994a1c8a5e3387c8a17'
MODEL_SHA256 = '652c3cd14e2fafd50c020e4a17b2cdd58bf0fa1e5b692a43f8dc8852270fc604'
ONNX = {
    'face_detection_yunet': ('face_detection_yunet_2023mar.onnx', '8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4'),
    'human_segmentation_pphumanseg': ('human_segmentation_pphumanseg_2023mar.onnx', '552d8a984054e59b5d773d24b9b12022b22046ceb2bbc4c9aaeaceb36a9ddf24'),
}


def setup():
    ensure_source(ROOT)
    repo = ROOT / 'third_party/APDrawingGAN'
    if not (repo / 'test.py').is_file():
        raise RuntimeError('Missing bundled third_party/APDrawingGAN source; restore the project checkout first.')
    target = repo / 'checkpoints/formal_author/300_net_gen.pt'
    if not target.is_file() or sha256(target) != MODEL_SHA256:
        local = ROOT / 'APDrawingGAN-Models1.zip'
        archive = local if local.is_file() and sha256(local) == ARCHIVE_SHA256 else ROOT / '.cache/models/APDrawingGAN-Models1.zip'
        download(MODEL_URL, archive, expected_sha256=ARCHIVE_SHA256)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix('.pt.download')
        try:
            with zipfile.ZipFile(archive) as bundle:
                # Extract only the exact member; never trust archive paths.
                with bundle.open('formal_author/300_net_gen.pt') as source, temporary.open('wb') as output:
                    shutil.copyfileobj(source, output)
            if sha256(temporary) != MODEL_SHA256:
                raise ValueError('Extracted APDrawingGAN checkpoint SHA256 mismatch')
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
    records = {'checkpoint': {'url': MODEL_URL, 'sha256': MODEL_SHA256}}
    for folder, (name, digest) in ONNX.items():
        url = f'https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/{folder}/{name}'
        download(url, ROOT / 'third_party/models' / name, expected_sha256=digest)
        records[name] = {'url': url, 'sha256': digest}
    (target.parent / 'provenance.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
    print(f'APDrawingGAN models ready: {target.parent}', flush=True)


if __name__ == '__main__':
    setup()
