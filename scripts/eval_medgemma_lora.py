"""Evaluate a MedGemma LoRA adapter on the final evaluation set."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import DB_PATH, LABELS_CSV, MEDGEMMA_ID, WARNING_TEXT  # noqa: E402
from src.db import fetch_runs, init_db, insert_run, register_model, register_prompt  # noqa: E402
from src.inference import run_single  # noqa: E402
from src.models import free_model, vram_used_gb  # noqa: E402
from src.postprocess import apply_uncertainty_rule, parse_json  # noqa: E402
from src.preprocessing import preprocess  # noqa: E402
from src.prompts import BASELINE  # noqa: E402


def load_cases(labels_csv: Path) -> List[Dict[str, Any]]:
    with labels_csv.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["true_label"] = row.get("label") or row.get("ground_truth") or None
        row["ground_truth"] = row["true_label"]
        row["image_path"] = (row.get("image_path") or row.get("file") or "").replace("\\", "/")
    return rows


def resolve_image_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (Path(__file__).resolve().parents[1] / path).resolve()


def existing_case_keys(db_path: Path) -> set[tuple[str, str]]:
    if not db_path.exists():
        return set()
    return {
        (str(row["case_id"]), str(row["model_name"]))
        for row in fetch_runs(db_path=db_path)
    }


def insert_failure_run(
    *,
    case: Dict[str, Any],
    model_id: int,
    prompt_id: int,
    config_name: str,
    error_message: str,
    db_path: Path,
) -> None:
    insert_run(
        case_id=case.get("case_id"),
        image_path=case.get("image_path"),
        true_label=case.get("true_label"),
        model_id=model_id,
        prompt_id=prompt_id,
        prompt_kind="lora_baseline",
        raw_output="",
        parsed_json={
            "image_quality": "insufficient",
            "predicted_class": "incertain",
            "confidence": 0.0,
            "visual_findings": [],
            "justification": f"Echec technique durant l'evaluation {config_name}.",
            "limitations": [error_message],
            "warning": WARNING_TEXT,
        },
        pred_class="incertain",
        confidence=0.0,
        ground_truth=case.get("true_label"),
        latency_ms=0,
        json_valid=False,
        reason="technical_error",
        warning_text=WARNING_TEXT,
        error_message=error_message,
        db_path=db_path,
    )


def load_lora_model(model_id: str, adapter: Path):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

    from src.config import HF_TOKEN

    auth = {"token": HF_TOKEN} if HF_TOKEN else {}
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    processor = AutoProcessor.from_pretrained(model_id, **auth)
    base_model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        quantization_config=bnb,
        device_map="auto",
        **auth,
    )
    model = PeftModel.from_pretrained(base_model, str(adapter))
    model.eval()
    return model, processor


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default="outputs/medgemma_lora_250")
    parser.add_argument("--config-name", default="medgemma_lora_250")
    parser.add_argument("--labels-csv", default=str(LABELS_CSV))
    parser.add_argument("--db-path", default=str(DB_PATH))
    parser.add_argument("--model-id", default=MEDGEMMA_ID)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--use-cached-runs", action="store_true")
    args = parser.parse_args()

    adapter = Path(args.adapter).resolve()
    labels_csv = Path(args.labels_csv).resolve()
    db_path = Path(args.db_path).resolve()
    if not adapter.exists():
        print(f"[ERREUR] adapter introuvable: {adapter}")
        return 1
    if not labels_csv.exists():
        print(f"[ERREUR] labels introuvables: {labels_csv}")
        return 1

    init_db(db_path)
    cases = load_cases(labels_csv)
    if args.max_cases is not None:
        cases = cases[: max(args.max_cases, 0)]
    cached = existing_case_keys(db_path) if args.use_cached_runs else set()

    model_name = args.config_name
    model_id = register_model(
        name=model_name,
        hf_id=f"{args.model_id}+adapter:{adapter.name}",
        quantization="4bit-nf4+lora",
        path=db_path,
    )
    prompt_id = register_prompt("lora_baseline", "v1", "single", BASELINE, path=db_path)

    print(f"[INFO] {len(cases)} cas charges depuis {labels_csv}")
    print(f"[load] base={args.model_id} adapter={adapter}")
    try:
        model, processor = load_lora_model(args.model_id, adapter)
        print(f"[VRAM] {vram_used_gb():.2f} Go")
    except Exception as exc:
        print(f"[ERREUR] chargement LoRA impossible: {exc}")
        return 1

    processed = 0
    skipped = 0
    technical_errors = 0
    for index, case in enumerate(cases, start=1):
        case_id = str(case["case_id"])
        if args.use_cached_runs and (case_id, model_name) in cached:
            skipped += 1
            continue
        processed += 1
        try:
            image_path = resolve_image_path(case["image_path"])
            image, qc = preprocess(image_path)
            raw, latency = run_single(model, processor, image, BASELINE)
            parsed, valid = parse_json(raw)
            final, reason = apply_uncertainty_rule(parsed, valid, qc["image_quality"])
            confidence = float(parsed.get("confidence")) if parsed and "confidence" in parsed else None
            insert_run(
                case_id=case_id,
                image_path=case["image_path"],
                true_label=case.get("true_label"),
                model_id=model_id,
                prompt_id=prompt_id,
                prompt_kind="lora_baseline",
                raw_output=raw,
                parsed_json=parsed,
                pred_class=final,
                confidence=confidence,
                ground_truth=case.get("true_label"),
                latency_ms=latency,
                json_valid=valid,
                reason=reason,
                warning_text=(parsed or {}).get("warning", WARNING_TEXT),
                db_path=db_path,
            )
            print(f"  [{index}/{len(cases)}] {case_id} -> {final} ({latency} ms, {reason})")
        except Exception as exc:
            technical_errors += 1
            insert_failure_run(
                case=case,
                model_id=model_id,
                prompt_id=prompt_id,
                config_name=model_name,
                error_message=str(exc),
                db_path=db_path,
            )
            print(f"  [{index}/{len(cases)}] {case_id} FAIL: {exc}")

    free_model(model, processor)
    print(f"[DONE] processed={processed} cached_skipped={skipped} technical_errors={technical_errors}")
    print(f"[DONE] runs sauvegardes dans {db_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
