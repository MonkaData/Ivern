"""Prépare un sous-échantillon RSNA équilibré en PNG pour l'évaluation.

Usage:
    python scripts/prepare_rsna.py ^
        --dicom-dir data/raw/stage_2_train_images ^
        --labels data/raw/stage_2_train_labels.csv ^
        --out-dir data/eval ^
        --n 30 ^
        --seed 42
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import DATA_EVAL, DATA_EVAL_IMAGES, DATA_RAW  # noqa: E402
from src.preprocessing import load_image  # noqa: E402
from src.rsna import (  # noqa: E402
    aggregate_rsna_labels,
    find_default_dicom_dir,
    find_default_labels_csv,
    index_dicoms,
    split_balanced,
)


def _resolve_default_path(value: str | None, fallback: Path | None, description: str) -> Path:
    if value:
        path = Path(value)
    elif fallback:
        path = fallback
    else:
        raise FileNotFoundError(f"{description} introuvable.")
    return path.resolve()


def _normalize_out_dir(path: Path) -> tuple[Path, Path]:
    out_dir = path.resolve()
    images_dir = out_dir / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    return out_dir, images_dir


def _copy_labels_if_needed(src_labels: Path) -> Path:
    target = (DATA_RAW / "stage_2_train_labels.csv").resolve()
    if src_labels.resolve() != target and not target.exists():
        DATA_RAW.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_labels, target)
    return target if target.exists() else src_labels.resolve()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dicom-dir", default=None, help="Dossier contenant les .dcm RSNA.")
    parser.add_argument("--labels", default=None, help="CSV RSNA stage_2_train_labels.csv.")
    parser.add_argument("--out-dir", default=str(DATA_EVAL), help="Dossier de sortie data/eval.")
    parser.add_argument("--n", type=int, default=30, help="Nombre total de cas à générer.")
    parser.add_argument("--seed", type=int, default=42, help="Seed d'échantillonnage.")
    args = parser.parse_args()

    try:
        dicom_dir = _resolve_default_path(
            args.dicom_dir,
            find_default_dicom_dir(),
            "Répertoire DICOM RSNA",
        )
        labels_csv = _resolve_default_path(
            args.labels,
            find_default_labels_csv(),
            "CSV des labels RSNA",
        )
    except FileNotFoundError as exc:
        print(f"[ERREUR] {exc}")
        print("[INFO] Utilise --dicom-dir et --labels si tes données sont ailleurs.")
        return 1

    if not dicom_dir.exists():
        print(f"[ERREUR] dossier DICOM introuvable: {dicom_dir}")
        return 1
    if not labels_csv.exists():
        print(f"[ERREUR] CSV labels introuvable: {labels_csv}")
        return 1

    out_dir, images_dir = _normalize_out_dir(Path(args.out_dir))
    labels_out = out_dir / "labels.csv"

    print(f"[INFO] labels CSV: {labels_csv}")
    print(f"[INFO] dossier DICOM: {dicom_dir}")
    print(f"[INFO] dossier sortie: {out_dir}")

    patients = aggregate_rsna_labels(labels_csv)
    positives = [patient for patient in patients if patient.target == 1]
    negatives = [patient for patient in patients if patient.target == 0]
    dicom_index = index_dicoms(dicom_dir)

    print(f"[INFO] patients lus: {len(patients)}")
    print(f"[INFO] positifs: {len(positives)}")
    print(f"[INFO] négatifs: {len(negatives)}")
    print(f"[INFO] DICOM trouvés: {len(dicom_index)}")

    selected, warnings = split_balanced(patients, n=args.n, seed=args.seed)
    for warning in warnings:
        print(f"[WARN] {warning}")

    rows = []
    missing_dicoms = 0
    generated = 0

    for index, patient in enumerate(selected, start=1):
        dicom_path = dicom_index.get(patient.patient_id)
        if dicom_path is None:
            print(f"[WARN] DICOM manquant pour patient_id={patient.patient_id}")
            missing_dicoms += 1
            continue

        case_id = f"case_{index:03d}"
        out_png = images_dir / f"{case_id}.png"
        try:
            image = load_image(dicom_path)
            image.save(out_png)
        except Exception as exc:
            print(f"[WARN] conversion DICOM -> PNG échouée pour {dicom_path}: {exc}")
            continue

        generated += 1
        rows.append(
            {
                "case_id": case_id,
                "image_path": str(out_png.relative_to(out_dir.parent.parent)),
                "patient_id": patient.patient_id,
                "target": patient.target,
                "label": patient.label,
                "ground_truth": patient.label,
                "original_dicom_path": str(dicom_path.relative_to(out_dir.parent.parent)),
                "has_bbox": int(patient.has_bbox),
                "bbox_count": patient.bbox_count,
            }
        )

    with labels_out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "case_id",
                "image_path",
                "patient_id",
                "target",
                "label",
                "ground_truth",
                "original_dicom_path",
                "has_bbox",
                "bbox_count",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    mirrored_labels = _copy_labels_if_needed(labels_csv)

    print(f"[INFO] DICOM manquants ignorés: {missing_dicoms}")
    print(f"[OK] PNG générés: {generated}")
    print(f"[OK] labels.csv: {labels_out}")
    print(f"[OK] labels source copiés ou disponibles: {mirrored_labels}")
    print(
        "[OK] répartition finale: "
        f"normal={sum(1 for row in rows if row['label'] == 'normal')} "
        f"suspicion_opacite={sum(1 for row in rows if row['label'] == 'suspicion_opacite')}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
