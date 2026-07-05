"""Smoke test : 1 image x 2 modèles. À exécuter sur la RTX 2080.

Usage:
    python scripts/smoke_inference.py data/eval/case_001.png
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import GEMMA3_ID, MEDGEMMA_ID  # noqa: E402
from src.inference import run_single  # noqa: E402
from src.models import free_model, load_model, vram_used_gb  # noqa: E402
from src.preprocessing import preprocess  # noqa: E402
from src.prompts import BASELINE  # noqa: E402


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python scripts/smoke_inference.py <image_path>")
        return 1
    image_path = sys.argv[1]
    img, qc = preprocess(image_path)
    print(f"[image] {img.size}  QC={qc}")

    for hf_id in (GEMMA3_ID, MEDGEMMA_ID):
        print(f"\n=== {hf_id} ===")
        try:
            model, processor = load_model(hf_id)
            print(f"  VRAM utilisée: {vram_used_gb():.2f} Go")
            raw, lat = run_single(model, processor, img, BASELINE)
            print(f"  latence: {lat} ms")
            print(f"  raw output:\n{raw[:1000]}")
        except Exception as e:
            print(f"  [ERREUR] {e}")
        finally:
            try:
                free_model(model, processor)  # type: ignore[name-defined]
            except Exception:
                pass
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
