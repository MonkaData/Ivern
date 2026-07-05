"""Loader 4-bit pour Gemma-3 / MedGemma + libération mémoire."""
from __future__ import annotations

import gc
from typing import Any, Tuple

from .config import HF_TOKEN


def _build_bnb_config():
    import torch
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )


def load_model(hf_id: str) -> Tuple[Any, Any]:
    """Charge un VLM (Gemma-3 ou MedGemma) en 4-bit nf4.

    Retourne (model, processor).
    """
    import torch  # noqa: F401
    from transformers import AutoProcessor

    try:
        from transformers import AutoModelForImageTextToText as AutoModel
    except ImportError:  # transformers < 4.45 fallback
        from transformers import AutoModelForCausalLM as AutoModel  # type: ignore

    bnb = _build_bnb_config()

    auth = {"token": HF_TOKEN} if HF_TOKEN else {}

    processor = AutoProcessor.from_pretrained(hf_id, **auth)
    model = AutoModel.from_pretrained(
        hf_id,
        quantization_config=bnb,
        device_map="auto",
        **auth,
    )
    model.eval()
    return model, processor


def free_model(*objs) -> None:
    """Libère les modèles et vide le cache CUDA."""
    for o in objs:
        try:
            del o
        except Exception:
            pass
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def vram_used_gb() -> float:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1e9
    except Exception:
        pass
    return 0.0
