"""Boucle d'évaluation : configurations x cas, avec mode secours."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import CONFIGS, DB_PATH, LABELS_CSV, WARNING_TEXT  # noqa: E402
from src.db import fetch_runs, init_db, insert_run, register_model, register_prompt  # noqa: E402
from src.inference import run_ensemble, run_single  # noqa: E402
from src.models import free_model, load_model, vram_used_gb  # noqa: E402
from src.postprocess import apply_uncertainty_rule, parse_json  # noqa: E402
from src.preprocessing import preprocess  # noqa: E402
from src.prompts import BASELINE, ENSEMBLE  # noqa: E402


def load_cases(labels_csv: Path) -> List[Dict[str, Any]]:
    if not labels_csv.exists():
        print(f"[ERREUR] labels manquants: {labels_csv}. Lance scripts/prepare_rsna.py d'abord.")
        return []
    with labels_csv.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["true_label"] = row.get("label") or row.get("ground_truth") or None
        row["ground_truth"] = row["true_label"]
        row["image_path"] = row.get("image_path") or row.get("file") or ""
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


def offline_summary(db_path: Path) -> int:
    if not db_path.exists():
        print(f"[ERREUR] mode offline impossible: base absente {db_path}")
        return 1
    runs = [dict(row) for row in fetch_runs(db_path=db_path)]
    print(f"[OK] mode offline: {len(runs)} runs déjà présents dans {db_path}")
    return 0


def register_config_entities(config_name: str, cfg: Dict[str, str], db_path: Path) -> tuple[int, int]:
    model_id = register_model(name=config_name, hf_id=cfg["hf_id"], quantization="4bit-nf4", path=db_path)
    if cfg["prompt_kind"] == "baseline":
        prompt_id = register_prompt("baseline", "v1", "single", BASELINE, path=db_path)
    else:
        prompt_id = register_prompt(
            "ensemble_reinforced",
            "v1",
            "ensemble",
            "\n\n--SEP--\n\n".join(text for _, text in ENSEMBLE),
            path=db_path,
        )
    return model_id, prompt_id


def insert_failure_run(
    *,
    case: Dict[str, Any],
    model_id: int,
    prompt_id: int,
    prompt_kind: str,
    error_message: str,
    db_path: Path,
) -> None:
    insert_run(
        case_id=case.get("case_id"),
        image_path=case.get("image_path"),
        true_label=case.get("true_label"),
        model_id=model_id,
        prompt_id=prompt_id,
        prompt_kind=prompt_kind,
        raw_output="",
        parsed_json={
            "image_quality": "insufficient",
            "predicted_class": "incertain",
            "confidence": 0.0,
            "visual_findings": [],
            "justification": "Échec technique durant l'évaluation.",
            "limitations": error_message,
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


def run_one_config(
    *,
    config_name: str,
    cases: List[Dict[str, Any]],
    db_path: Path,
    use_cached_runs: bool,
) -> dict[str, int]:
    cfg = CONFIGS[config_name]
    stats = {"processed": 0, "cached_skipped": 0, "technical_errors": 0}
    print(f"\n========== {config_name} :: {cfg['label']} ==========")

    model_id, prompt_id = register_config_entities(config_name, cfg, db_path)
    cached = existing_case_keys(db_path) if use_cached_runs else set()

    model = processor = None
    model_load_error = None
    try:
        print(f"[load] {cfg['hf_id']} (4-bit nf4)")
        model, processor = load_model(cfg["hf_id"])
        print(f"[VRAM] {vram_used_gb():.2f} Go")
    except Exception as exc:
        model_load_error = str(exc)
        print(f"[WARN] chargement modèle impossible: {exc}")

    for index, case in enumerate(cases, start=1):
        case_id = str(case["case_id"])
        if use_cached_runs and (case_id, config_name) in cached:
            stats["cached_skipped"] += 1
            continue

        image_path = resolve_image_path(case["image_path"])
        ground_truth = case.get("true_label")
        stats["processed"] += 1

        if model_load_error is not None:
            insert_failure_run(
                case=case,
                model_id=model_id,
                prompt_id=prompt_id,
                prompt_kind=cfg["prompt_kind"],
                error_message=f"model_load_failed: {model_load_error}",
                db_path=db_path,
            )
            stats["technical_errors"] += 1
            print(f"  [{index}/{len(cases)}] {case_id} -> incertain (model_load_failed)")
            continue

        try:
            image, qc = preprocess(image_path)
        except Exception as exc:
            insert_failure_run(
                case=case,
                model_id=model_id,
                prompt_id=prompt_id,
                prompt_kind=cfg["prompt_kind"],
                error_message=f"preprocess_failed: {exc}",
                db_path=db_path,
            )
            stats["technical_errors"] += 1
            print(f"  [{index}/{len(cases)}] {case_id} preprocess FAIL: {exc}")
            continue

        try:
            if cfg["prompt_kind"] == "baseline":
                raw, latency = run_single(model, processor, image, BASELINE)
                parsed, valid = parse_json(raw)
                final, reason = apply_uncertainty_rule(parsed, valid, qc["image_quality"])
                confidence = float(parsed.get("confidence")) if parsed and "confidence" in parsed else None
                insert_run(
                    case_id=case_id,
                    image_path=case["image_path"],
                    true_label=ground_truth,
                    model_id=model_id,
                    prompt_id=prompt_id,
                    prompt_kind=cfg["prompt_kind"],
                    raw_output=raw,
                    parsed_json=parsed,
                    pred_class=final,
                    confidence=confidence,
                    ground_truth=ground_truth,
                    latency_ms=latency,
                    json_valid=valid,
                    reason=reason,
                    warning_text=(parsed or {}).get("warning", WARNING_TEXT),
                    db_path=db_path,
                )
                print(f"  [{index}/{len(cases)}] {case_id} -> {final} ({latency} ms, {reason})")
            else:
                ensemble_results = run_ensemble(model, processor, image, ENSEMBLE)
                votes = []
                parsed_each = []
                total_latency = 0
                raw_payload = []
                for prompt_name, raw_text, latency in ensemble_results:
                    parsed_i, valid_i = parse_json(raw_text)
                    votes.append(parsed_i["predicted_class"] if valid_i and parsed_i else "incertain")
                    parsed_each.append(
                        {
                            "prompt": prompt_name,
                            "parsed": parsed_i,
                            "valid": valid_i,
                            "latency_ms": latency,
                        }
                    )
                    raw_payload.append({"prompt": prompt_name, "raw_output": raw_text})
                    total_latency += latency

                primary = next((item["parsed"] for item in parsed_each if item["valid"]), None)
                valid = primary is not None
                final, reason = apply_uncertainty_rule(
                    primary,
                    valid,
                    qc["image_quality"],
                    ensemble_classes=votes,
                )
                confidence = float(primary.get("confidence")) if primary and "confidence" in primary else None
                parsed_json = {
                    "ensemble_votes": votes,
                    "primary": primary,
                    "qc": qc,
                    "warning": WARNING_TEXT,
                }
                insert_run(
                    case_id=case_id,
                    image_path=case["image_path"],
                    true_label=ground_truth,
                    model_id=model_id,
                    prompt_id=prompt_id,
                    prompt_kind=cfg["prompt_kind"],
                    raw_output=json.dumps(raw_payload, ensure_ascii=False),
                    parsed_json=parsed_json,
                    pred_class=final,
                    confidence=confidence,
                    ground_truth=ground_truth,
                    latency_ms=total_latency,
                    json_valid=valid,
                    reason=reason,
                    warning_text=WARNING_TEXT,
                    db_path=db_path,
                )
                print(f"  [{index}/{len(cases)}] {case_id} -> {final} votes={votes}")
        except Exception as exc:
            insert_failure_run(
                case=case,
                model_id=model_id,
                prompt_id=prompt_id,
                prompt_kind=cfg["prompt_kind"],
                error_message=f"inference_failed: {exc}",
                db_path=db_path,
            )
            stats["technical_errors"] += 1
            print(f"  [{index}/{len(cases)}] {case_id} INFER FAIL: {exc}")

    if model is not None or processor is not None:
        free_model(model, processor)
        print(f"[free] mémoire libérée. VRAM max: {vram_used_gb():.2f} Go")
    return stats


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels-csv", default=str(LABELS_CSV), help="Chemin du labels.csv évalué.")
    parser.add_argument("--db-path", default=str(DB_PATH), help="Chemin de sortie SQLite.")
    parser.add_argument("--offline", action="store_true", help="Ne lance aucune inférence, utilise la base existante.")
    parser.add_argument("--reset-db", action="store_true", help="Supprime la base cible avant de relancer l'évaluation.")
    parser.add_argument(
        "--configs",
        nargs="+",
        default=list(CONFIGS.keys()),
        choices=list(CONFIGS.keys()),
        help="Sous-ensemble de configurations à évaluer.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Nombre maximum de cas à traiter depuis labels.csv.",
    )
    parser.add_argument(
        "--use-cached-runs",
        action="store_true",
        help="Ne relance pas les couples (case_id, configuration) déjà présents.",
    )
    args = parser.parse_args()

    labels_csv = Path(args.labels_csv).resolve()
    db_path = Path(args.db_path).resolve()
    if args.reset_db and db_path.exists():
        db_path.unlink()
    init_db(db_path)

    if args.offline:
        return offline_summary(db_path)

    cases = load_cases(labels_csv)
    if not cases:
        return 1
    if args.max_cases is not None:
        cases = cases[: max(args.max_cases, 0)]

    print(f"[INFO] {len(cases)} cas chargés depuis {labels_csv}")
    global_stats = {"processed": 0, "cached_skipped": 0, "technical_errors": 0}
    for config_name in args.configs:
        stats = run_one_config(
            config_name=config_name,
            cases=cases,
            db_path=db_path,
            use_cached_runs=args.use_cached_runs,
        )
        for key, value in stats.items():
            global_stats[key] += value

    print(f"\n[DONE] runs sauvegardés dans {db_path}")
    print(
        "[DONE] résumé: "
        f"processed={global_stats['processed']} "
        f"cached_skipped={global_stats['cached_skipped']} "
        f"technical_errors={global_stats['technical_errors']}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
