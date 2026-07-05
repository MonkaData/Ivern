from scripts.export_report import select_commented_cases


def _run(idx, case_id, model_name, pred, gt, json_valid=1, error_message=""):
    return {
        "id": idx,
        "case_id": case_id,
        "model_name": model_name,
        "pred_class": pred,
        "ground_truth": gt,
        "json_valid": json_valid,
        "error_message": error_message,
    }


def test_select_commented_cases_balances_categories():
    rows = [
        _run(1, "c1", "m", "normal", "suspicion_opacite"),
        _run(2, "c2", "m", "suspicion_opacite", "normal"),
        _run(3, "c3", "m", "incertain", "normal"),
        _run(4, "c4", "m", "incertain", "normal", json_valid=0),
        _run(5, "c5", "m", "incertain", "normal", error_message="fail"),
        _run(6, "c6", "m", "normal", "normal"),
        _run(7, "c7", "m", "suspicion_opacite", "suspicion_opacite"),
    ]
    selected = select_commented_cases(rows, target_count=6)
    assert len(selected) == 6
    case_ids = {row["case_id"] for row in selected}
    assert {"c1", "c2", "c3", "c4", "c5"}.issubset(case_ids)
