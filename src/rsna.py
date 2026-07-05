"""Helpers pour organiser et agréger le dataset RSNA Pneumonia."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .config import DATA_RAW


@dataclass(frozen=True)
class RsnaPatient:
    patient_id: str
    target: int
    label: str
    has_bbox: bool
    bbox_count: int


def target_to_label(target: int) -> str:
    return "suspicion_opacite" if int(target) == 1 else "normal"


def aggregate_rsna_labels(csv_path: Path) -> List[RsnaPatient]:
    """Agrège les lignes RSNA au niveau patient."""
    aggregated: Dict[str, Dict[str, int]] = {}
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            patient_id = (
                row.get("patientId")
                or row.get("PatientID")
                or row.get("patient_id")
            )
            target_raw = row.get("Target") or row.get("target")
            if not patient_id or target_raw is None:
                continue

            target = int(target_raw)
            bbox_fields = [row.get("x"), row.get("y"), row.get("width"), row.get("height")]
            has_bbox = target == 1 and all(v not in (None, "") for v in bbox_fields)

            item = aggregated.setdefault(
                patient_id,
                {"target": 0, "bbox_count": 0},
            )
            item["target"] = max(item["target"], target)
            if has_bbox:
                item["bbox_count"] += 1

    patients = [
        RsnaPatient(
            patient_id=patient_id,
            target=values["target"],
            label=target_to_label(values["target"]),
            has_bbox=values["bbox_count"] > 0,
            bbox_count=values["bbox_count"],
        )
        for patient_id, values in aggregated.items()
    ]
    patients.sort(key=lambda item: item.patient_id)
    return patients


def find_default_labels_csv() -> Optional[Path]:
    candidates = [
        DATA_RAW / "stage_2_train_labels.csv",
        DATA_RAW.parent / "stage_2_train_labels.csv",
        Path("stage_2_train_labels.csv"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def find_default_dicom_dir() -> Optional[Path]:
    candidates = [
        DATA_RAW / "stage_2_train_images",
        Path("Nouveau dossier"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def index_dicoms(dicom_dir: Path) -> Dict[str, Path]:
    """Retourne {patient_id: absolute_path}."""
    return {path.stem: path.resolve() for path in dicom_dir.rglob("*.dcm")}


def split_balanced(
    patients: Iterable[RsnaPatient],
    *,
    n: int,
    seed: int,
) -> tuple[List[RsnaPatient], List[str]]:
    import random

    warnings: List[str] = []
    rng = random.Random(seed)
    positives = [p for p in patients if p.target == 1]
    negatives = [p for p in patients if p.target == 0]
    rng.shuffle(positives)
    rng.shuffle(negatives)

    wanted_pos = n // 2
    wanted_neg = n - wanted_pos
    take_pos = min(wanted_pos, len(positives))
    take_neg = min(wanted_neg, len(negatives))

    if take_pos < wanted_pos or take_neg < wanted_neg:
        warnings.append(
            "Échantillon parfaitement équilibré impossible: "
            f"demandé pos={wanted_pos}/neg={wanted_neg}, "
            f"disponible pos={len(positives)}/neg={len(negatives)}."
        )

    selected = positives[:take_pos] + negatives[:take_neg]
    remaining = [p for p in positives[take_pos:] + negatives[take_neg:]]
    rng.shuffle(remaining)
    while len(selected) < min(n, len(positives) + len(negatives)) and remaining:
        selected.append(remaining.pop())

    rng.shuffle(selected)
    return selected, warnings
