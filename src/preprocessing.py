"""Préprocessing image : DICOM/PNG -> RGB 896 + Quality Control."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple, Dict, Any, Union

import numpy as np
from PIL import Image, ImageOps

from .config import (
    IMAGE_LONG_SIDE,
    QC_LUMINANCE_STD_MIN,
    QC_LUMINANCE_MEAN_RANGE,
)

PathLike = Union[str, Path]


def _load_dicom(path: Path) -> Image.Image:
    """Charge un DICOM et retourne une PIL Image en mode L (8-bit)."""
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_voi_lut

    ds = pydicom.dcmread(str(path))
    arr = apply_voi_lut(ds.pixel_array, ds).astype(np.float32)

    slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
    arr = arr * slope + intercept

    # Robust normalization 1..99 percentile to preserve chest contrast.
    low = float(np.percentile(arr, 1))
    high = float(np.percentile(arr, 99))
    if high <= low:
        low = float(arr.min())
        high = float(arr.max())
    if high > low:
        arr = np.clip(arr, low, high)
        arr = (arr - low) / max(high - low, 1e-6) * 255.0
    else:
        arr = np.zeros_like(arr)
    arr = arr.astype(np.uint8)

    # Some DICOMs have inverted photometric interpretation
    photometric = getattr(ds, "PhotometricInterpretation", "")
    if photometric == "MONOCHROME1":
        arr = 255 - arr

    return Image.fromarray(arr, mode="L")


def load_image(path: PathLike) -> Image.Image:
    """Charge un fichier PNG/JPG/DICOM et retourne une PIL.Image en RGB carré 896."""
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in (".dcm", ".dicom"):
        img = _load_dicom(p)
    else:
        img = Image.open(p)
        if img.mode != "RGB":
            img = img.convert("L")  # cxr -> grayscale d'abord
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = _resize_and_pad(img, IMAGE_LONG_SIDE)
    return img


def _resize_and_pad(img: Image.Image, target: int) -> Image.Image:
    """Redimensionne côté long = target, puis pad noir pour avoir un carré."""
    w, h = img.size
    if w >= h:
        new_w = target
        new_h = int(round(h * target / w))
    else:
        new_h = target
        new_w = int(round(w * target / h))
    img = img.resize((new_w, new_h), Image.LANCZOS)
    pad_w = target - new_w
    pad_h = target - new_h
    padding = (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2)
    img = ImageOps.expand(img, border=padding, fill=(0, 0, 0))
    return img


def quality_check(image: Image.Image) -> Dict[str, Any]:
    """QC simple basé sur luminance moyenne / écart-type."""
    arr = np.asarray(image.convert("L"), dtype=np.float32)
    mean = float(arr.mean())
    std = float(arr.std())

    if std < QC_LUMINANCE_STD_MIN:
        quality = "insufficient"
    elif not (QC_LUMINANCE_MEAN_RANGE[0] <= mean <= QC_LUMINANCE_MEAN_RANGE[1]):
        quality = "degraded"
    else:
        quality = "ok"

    return {
        "image_quality": quality,
        "luminance_mean": round(mean, 2),
        "luminance_std": round(std, 2),
    }


def preprocess(path: PathLike) -> Tuple[Image.Image, Dict[str, Any]]:
    """Pipeline complet : load + QC. Retourne (PIL.Image, qc_dict)."""
    img = load_image(path)
    qc = quality_check(img)
    return img, qc


if __name__ == "__main__":  # pragma: no cover
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.preprocessing <image_path>")
        sys.exit(1)
    img, qc = preprocess(sys.argv[1])
    print(f"Image: {img.size} mode={img.mode}")
    print(f"QC   : {qc}")
