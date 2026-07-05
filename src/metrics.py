"""Calcul de métriques : accuracy, macro-F1, sensibilité, spécificité, latence."""
from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from .config import CLASSES
from .postprocess import auto_error_type


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def summarize_errors(runs: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    summary = {
        "false_positives": 0,
        "false_negatives": 0,
        "uncertain_cases": 0,
        "json_invalid_cases": 0,
        "technical_errors": 0,
        "hallucination_flags": 0,
    }
    for run in runs:
        pred = run.get("pred_class")
        ground_truth = run.get("ground_truth")
        json_valid = bool(run.get("json_valid"))
        if run.get("error_message"):
            summary["technical_errors"] += 1

        error_kind = auto_error_type(pred, ground_truth, json_valid)
        if error_kind == "faux_positif":
            summary["false_positives"] += 1
        elif error_kind == "faux_negatif":
            summary["false_negatives"] += 1
        elif error_kind == "incertain_a_analyser":
            summary["uncertain_cases"] += 1
        elif error_kind == "json_invalide":
            summary["json_invalid_cases"] += 1

        comment_blob = " ".join(
            str(run.get(key, "") or "")
            for key in ("error_type", "comment", "reason")
        ).lower()
        if "hallucin" in comment_blob:
            summary["hallucination_flags"] += 1
    return summary


def compute_metrics(runs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Calcule les métriques principales sur un set de runs."""
    n = len(runs)
    if n == 0:
        return {"n": 0}

    correct = sum(1 for r in runs if r["pred_class"] == r["ground_truth"])
    accuracy = correct / n

    decided = [r for r in runs if r["pred_class"] != "incertain"]
    accuracy_decided = (
        sum(1 for r in decided if r["pred_class"] == r["ground_truth"]) / len(decided)
        if decided else 0.0
    )

    positives = [r for r in runs if r["ground_truth"] == "suspicion_opacite"]
    true_positives = sum(1 for r in positives if r["pred_class"] == "suspicion_opacite")
    sensitivity = _safe_div(true_positives, len(positives))

    negatives = [r for r in runs if r["ground_truth"] == "normal"]
    true_negatives = sum(1 for r in negatives if r["pred_class"] == "normal")
    specificity = _safe_div(true_negatives, len(negatives))

    f1s = []
    for cls in ("normal", "suspicion_opacite"):
        tp = sum(1 for r in runs if r["pred_class"] == cls and r["ground_truth"] == cls)
        fp = sum(1 for r in runs if r["pred_class"] == cls and r["ground_truth"] != cls)
        fn = sum(1 for r in runs if r["pred_class"] != cls and r["ground_truth"] == cls)
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1s.append(_safe_div(2 * precision * recall, precision + recall))
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0

    json_valid_pct = sum(1 for r in runs if r["json_valid"]) / n * 100.0
    uncertain_pct = sum(1 for r in runs if r["pred_class"] == "incertain") / n * 100.0

    latencies = [r["latency_ms"] for r in runs if r.get("latency_ms") is not None]
    if latencies:
        sorted_latencies = sorted(latencies)
        latency_mean = float(np.mean(sorted_latencies))
        p50 = float(np.percentile(sorted_latencies, 50))
        p95 = float(np.percentile(sorted_latencies, 95))
    else:
        latency_mean = p50 = p95 = 0.0

    error_summary = summarize_errors(runs)
    return {
        "n": n,
        "accuracy": round(accuracy, 4),
        "accuracy_decided": round(accuracy_decided, 4),
        "macro_f1": round(macro_f1, 4),
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "json_valid_pct": round(json_valid_pct, 2),
        "uncertain_pct": round(uncertain_pct, 2),
        "latency_mean_ms": round(latency_mean, 1),
        "latency_p50_ms": round(p50, 1),
        "latency_p95_ms": round(p95, 1),
        **error_summary,
    }


def confusion_matrix(runs: Sequence[Dict[str, Any]]) -> Tuple[np.ndarray, List[str]]:
    """Matrice de confusion 3x3 (lignes=GT, colonnes=Pred)."""
    classes = list(CLASSES)
    idx = {cls: i for i, cls in enumerate(classes)}
    matrix = np.zeros((3, 3), dtype=int)
    for run in runs:
        gt = run.get("ground_truth")
        pred = run.get("pred_class")
        if gt in idx and pred in idx:
            matrix[idx[gt], idx[pred]] += 1
    return matrix, classes
