import tempfile
from pathlib import Path

from src import db


def test_init_and_insert():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "t.sqlite"
        db.init_db(p)
        m = db.register_model("baseline", "google/x", "4bit-nf4", path=p)
        pr = db.register_prompt("base", "v1", "single", "hello", path=p)
        rid = db.insert_run(
            case_id="c1", image_path="i.png", model_id=m, prompt_id=pr,
            true_label="normal", prompt_kind="baseline",
            raw_output="{}", parsed_json={"a": 1},
            pred_class="normal", confidence=0.8, ground_truth="normal",
            latency_ms=100, json_valid=True, warning_text="warn", db_path=p,
        )
        db.insert_error(rid, "FP", "test", db_path=p)
        runs = db.fetch_runs(db_path=p)
        assert len(runs) == 1
        assert runs[0]["true_label"] == "normal"
        assert runs[0]["prompt_kind"] == "baseline"
        errs = db.fetch_errors(db_path=p)
        assert len(errs) == 1
