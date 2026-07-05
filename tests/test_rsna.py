import csv
from pathlib import Path

from src.rsna import aggregate_rsna_labels, split_balanced


def test_aggregate_rsna_labels_counts_bbox(tmp_path: Path):
    csv_path = tmp_path / "labels.csv"
    csv_path.write_text(
        "patientId,x,y,width,height,Target\n"
        "p1,1,2,3,4,1\n"
        "p1,5,6,7,8,1\n"
        "p2,,,,,0\n",
        encoding="utf-8",
    )

    patients = aggregate_rsna_labels(csv_path)
    assert len(patients) == 2
    p1 = next(patient for patient in patients if patient.patient_id == "p1")
    p2 = next(patient for patient in patients if patient.patient_id == "p2")
    assert p1.target == 1
    assert p1.label == "suspicion_opacite"
    assert p1.has_bbox is True
    assert p1.bbox_count == 2
    assert p2.target == 0
    assert p2.label == "normal"
    assert p2.bbox_count == 0


def test_split_balanced_warns_when_unbalanced():
    patients = [
        type("Patient", (), {"target": 1, "patient_id": "p1"})(),
        type("Patient", (), {"target": 0, "patient_id": "n1"})(),
        type("Patient", (), {"target": 0, "patient_id": "n2"})(),
    ]
    selected, warnings = split_balanced(patients, n=4, seed=42)
    assert len(selected) == 3
    assert warnings
