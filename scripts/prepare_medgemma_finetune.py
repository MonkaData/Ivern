"""Prepare a small balanced RSNA subset for MedGemma LoRA experiments.

This script intentionally excludes the current evaluation cases by patient_id so
the 30-case report can remain an independent final test.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import WARNING_TEXT  # noqa: E402
from src.preprocessing import load_image  # noqa: E402
from src.rsna import (  # noqa: E402
    RsnaPatient,
    aggregate_rsna_labels,
    find_default_dicom_dir,
    find_default_labels_csv,
    index_dicoms,
)


def _rel_posix(path: Path) -> str:
    return path.relative_to(Path.cwd()).as_posix()


PROMPT = (
    "Analyse cette radiographie thoracique frontale dans un cadre pedagogique. "
    "Retourne uniquement un JSON valide avec les cles image_quality, predicted_class, "
    "confidence, visual_findings, justification, limitations et warning. "
    "Les classes autorisees sont normal, suspicion_opacite et incertain."
)


def _read_excluded_patients(labels_csv: Path | None) -> set[str]:
    if labels_csv is None or not labels_csv.exists():
        return set()
    with labels_csv.open("r", newline="", encoding="utf-8") as handle:
        return {
            row["patient_id"]
            for row in csv.DictReader(handle)
            if row.get("patient_id")
        }


def _balanced_sample(patients: Iterable[RsnaPatient], n: int, seed: int) -> list[RsnaPatient]:
    rng = random.Random(seed)
    positives = [p for p in patients if p.target == 1]
    negatives = [p for p in patients if p.target == 0]
    rng.shuffle(positives)
    rng.shuffle(negatives)
    half = n // 2
    selected = positives[:half] + negatives[: n - half]
    rng.shuffle(selected)
    return selected


def _split(rows: list[dict], seed: int) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    shuffled = rows[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)
    return {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }


def _target_json(label: str) -> str:
    if label == "suspicion_opacite":
        visual = ["label RSNA positif pour opacite pulmonaire"]
        justification = "Le cas est annote comme suspicion d'opacite dans le jeu RSNA."
        confidence = 0.85
    else:
        visual = ["label RSNA negatif pour opacite pulmonaire"]
        justification = "Le cas est annote comme normal dans le jeu RSNA."
        confidence = 0.85
    payload = {
        "image_quality": "ok",
        "predicted_class": label,
        "confidence": confidence,
        "visual_findings": visual,
        "justification": justification,
        "limitations": [
            "Supervision issue des labels RSNA, pas d'annotation clinique complete.",
            "Prototype pedagogique non destine au diagnostic.",
        ],
        "warning": WARNING_TEXT,
    }
    return json.dumps(payload, ensure_ascii=False)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            item = {
                "case_id": row["case_id"],
                "image_path": row["image_path"],
                "label": row["label"],
                "messages": [
                    {"role": "user", "content": PROMPT},
                    {"role": "assistant", "content": _target_json(row["label"])},
                ],
            }
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dicom-dir", default=None)
    parser.add_argument("--labels", default=None)
    parser.add_argument("--exclude-labels", default="data/eval/labels.csv")
    parser.add_argument("--out-dir", default="data/finetune_medgemma_250")
    parser.add_argument("--n", type=int, default=250)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    dicom_dir = Path(args.dicom_dir).resolve() if args.dicom_dir else find_default_dicom_dir()
    labels_csv = Path(args.labels).resolve() if args.labels else find_default_labels_csv()
    if dicom_dir is None or not dicom_dir.exists():
        raise FileNotFoundError("DICOM directory not found. Use --dicom-dir.")
    if labels_csv is None or not labels_csv.exists():
        raise FileNotFoundError("RSNA labels CSV not found. Use --labels.")

    out_dir = Path(args.out_dir).resolve()
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    excluded = _read_excluded_patients(Path(args.exclude_labels) if args.exclude_labels else None)
    dicom_index = index_dicoms(dicom_dir)
    patients = [
        patient
        for patient in aggregate_rsna_labels(labels_csv)
        if patient.patient_id not in excluded and patient.patient_id in dicom_index
    ]
    selected = _balanced_sample(patients, args.n, args.seed)

    rows: list[dict] = []
    for index, patient in enumerate(selected, start=1):
        dicom_path = dicom_index[patient.patient_id]
        case_id = f"ft_{index:04d}"
        out_png = images_dir / f"{case_id}.png"
        image = load_image(dicom_path)
        image.save(out_png)
        rows.append(
            {
                "case_id": case_id,
                "image_path": _rel_posix(out_png),
                "patient_id": patient.patient_id,
                "target": patient.target,
                "label": patient.label,
                "ground_truth": patient.label,
                "original_dicom_path": _rel_posix(dicom_path),
                "has_bbox": int(patient.has_bbox),
                "bbox_count": patient.bbox_count,
            }
        )

    with (out_dir / "labels.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    splits = _split(rows, args.seed)
    for name, split_rows in splits.items():
        _write_jsonl(out_dir / f"{name}.jsonl", split_rows)

    print(f"[OK] out_dir={out_dir}")
    print(f"[OK] total={len(rows)} train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")
    print(
        "[OK] distribution: "
        f"normal={sum(1 for row in rows if row['label'] == 'normal')} "
        f"suspicion_opacite={sum(1 for row in rows if row['label'] == 'suspicion_opacite')}"
    )
    print(f"[OK] excluded final eval patients={len(excluded)}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
