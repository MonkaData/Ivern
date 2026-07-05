from src.metrics import compute_metrics, confusion_matrix, summarize_errors


def make(preds, gts, confs=None, lats=None, valids=None):
    n = len(preds)
    confs = confs or [0.9] * n
    lats = lats or [100] * n
    valids = valids or [1] * n
    return [
        {
            "pred_class": p, "ground_truth": g, "confidence": c,
            "latency_ms": l, "json_valid": v,
        }
        for p, g, c, l, v in zip(preds, gts, confs, lats, valids)
    ]


def test_perfect_classifier():
    runs = make(
        ["normal"] * 5 + ["suspicion_opacite"] * 5,
        ["normal"] * 5 + ["suspicion_opacite"] * 5,
    )
    m = compute_metrics(runs)
    assert m["accuracy"] == 1.0
    assert m["sensitivity"] == 1.0
    assert m["specificity"] == 1.0
    assert m["macro_f1"] == 1.0


def test_all_uncertain():
    runs = make(["incertain"] * 4, ["normal", "normal", "suspicion_opacite", "suspicion_opacite"])
    m = compute_metrics(runs)
    assert m["uncertain_pct"] == 100.0
    assert m["sensitivity"] == 0.0
    assert m["specificity"] == 0.0


def test_specificity_counts_uncertain_as_not_true_negative():
    runs = make(["normal", "incertain"], ["normal", "normal"])
    m = compute_metrics(runs)
    assert m["specificity"] == 0.5


def test_confusion_matrix_shape():
    runs = make(["normal", "suspicion_opacite", "incertain"],
                ["normal", "normal", "suspicion_opacite"])
    M, classes = confusion_matrix(runs)
    assert M.shape == (3, 3)
    assert classes == ["normal", "suspicion_opacite", "incertain"]
    assert M[0, 0] == 1  # normal -> normal
    assert M[0, 1] == 1  # normal -> suspicion_opacite (FP)
    assert M[1, 2] == 1  # suspicion_opacite -> incertain


def test_error_summary_counts():
    runs = [
        {"pred_class": "suspicion_opacite", "ground_truth": "normal", "json_valid": 1, "latency_ms": 10},
        {"pred_class": "normal", "ground_truth": "suspicion_opacite", "json_valid": 1, "latency_ms": 10},
        {"pred_class": "incertain", "ground_truth": "normal", "json_valid": 1, "latency_ms": 10},
        {"pred_class": "incertain", "ground_truth": "normal", "json_valid": 0, "latency_ms": 10},
        {
            "pred_class": "normal",
            "ground_truth": "normal",
            "json_valid": 1,
            "latency_ms": 10,
            "comment": "hallucination signalée",
        },
        {
            "pred_class": "normal",
            "ground_truth": "normal",
            "json_valid": 1,
            "latency_ms": 10,
            "error_message": "boom",
        },
    ]
    summary = summarize_errors(runs)
    assert summary["false_positives"] == 1
    assert summary["false_negatives"] == 1
    assert summary["uncertain_cases"] == 1
    assert summary["json_invalid_cases"] == 1
    assert summary["hallucination_flags"] == 1
    assert summary["technical_errors"] == 1
