import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from robot_portrait.lineart import portrait_lineart

def test_lineart_shape():
    img = np.full((64, 64, 3), 255, np.uint8)
    out = portrait_lineart(img, max_side=64)
    assert out.shape == (64, 64)
