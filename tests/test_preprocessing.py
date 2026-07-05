import numpy as np
from PIL import Image

from src.preprocessing import quality_check, _resize_and_pad


def test_resize_and_pad_makes_square():
    img = Image.new("RGB", (1024, 600), (128, 128, 128))
    out = _resize_and_pad(img, 896)
    assert out.size == (896, 896)


def test_qc_black_image_insufficient():
    img = Image.new("RGB", (896, 896), (0, 0, 0))
    qc = quality_check(img)
    assert qc["image_quality"] == "insufficient"


def test_qc_normal_image_ok():
    arr = (np.random.default_rng(0).normal(128, 50, (896, 896))).clip(0, 255).astype("uint8")
    img = Image.fromarray(arr).convert("RGB")
    qc = quality_check(img)
    assert qc["image_quality"] == "ok"
