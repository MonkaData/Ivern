"""Inférence : run_single (1 prompt) + run_ensemble (N prompts)."""
from __future__ import annotations

import time
from typing import Any, List, Tuple

from PIL import Image

from .config import MAX_NEW_TOKENS


def _build_messages(image: Image.Image, prompt: str) -> list:
    """Construit la structure messages multimodale style Gemma-3 / MedGemma."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def run_single(model: Any, processor: Any, image: Image.Image, prompt: str) -> Tuple[str, int]:
    """Exécute une inférence et retourne (raw_text, latency_ms)."""
    import torch

    messages = _build_messages(image, prompt)
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    # déplacer sur le device du modèle
    device = next(model.parameters()).device
    inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[-1]

    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )
    latency_ms = int((time.perf_counter() - t0) * 1000)

    gen = out[0][input_len:]
    text = processor.decode(gen, skip_special_tokens=True)
    return text.strip(), latency_ms


def run_ensemble(
    model: Any,
    processor: Any,
    image: Image.Image,
    prompts: List[Tuple[str, str]],
) -> List[Tuple[str, str, int]]:
    """Exécute N prompts. Retourne [(prompt_name, raw_text, latency_ms), ...]."""
    results: List[Tuple[str, str, int]] = []
    for name, prompt in prompts:
        raw, lat = run_single(model, processor, image, prompt)
        results.append((name, raw, lat))
    return results
