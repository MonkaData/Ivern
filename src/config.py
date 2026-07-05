"""Configuration centrale du projet."""
from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

# ---------- Paths ----------
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_EVAL = ROOT / "data" / "eval"
DATA_EVAL_IMAGES = DATA_EVAL / "images"
LABELS_CSV = DATA_EVAL / "labels.csv"
DB_PATH = ROOT / "runs.sqlite"
REPORT_DIR = ROOT / "reports"

DATA_EVAL.mkdir(parents=True, exist_ok=True)
DATA_EVAL_IMAGES.mkdir(parents=True, exist_ok=True)
DATA_RAW.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Modèles ----------
HF_TOKEN = os.environ.get("HF_TOKEN", "").strip() or None

GEMMA3_ID = "google/gemma-3-4b-it"
MEDGEMMA_ID = "google/medgemma-4b-it"

# Configurations comparées
CONFIGS = {
    "gemma3_baseline": {
        "label": "Gemma-3 4B (baseline)",
        "hf_id": GEMMA3_ID,
        "prompt_kind": "baseline",
    },
    "medgemma_baseline": {
        "label": "MedGemma 4B (baseline)",
        "hf_id": MEDGEMMA_ID,
        "prompt_kind": "baseline",
    },
    "medgemma_ensemble_reinforced": {
        "label": "MedGemma 4B (ensemble + prompts renforcés)",
        "hf_id": MEDGEMMA_ID,
        "prompt_kind": "ensemble",
    },
}

# ---------- Image / inference ----------
IMAGE_LONG_SIDE = 896
QC_LUMINANCE_STD_MIN = 15.0
QC_LUMINANCE_MEAN_RANGE = (20.0, 235.0)

CONF_THRESHOLD = 0.60
MAX_NEW_TOKENS = 512

# ---------- Classes ----------
CLASSES = ("normal", "suspicion_opacite", "incertain")
WARNING_TEXT = (
    "Prototype pédagogique. Non destiné au diagnostic. "
    "Validation par un professionnel qualifié requise."
)
