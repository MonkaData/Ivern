import json

from src.postprocess import apply_uncertainty_rule, parse_json


VALID = {
    "image_quality": "ok",
    "predicted_class": "normal",
    "confidence": 0.85,
    "visual_findings": ["champs clairs"],
    "justification": "Pas d'opacité visible.",
    "limitations": "Vue unique frontale.",
    "warning": "ignored",
}


def test_parse_clean_json():
    parsed, valid = parse_json(json.dumps(VALID))
    assert valid is True
    assert parsed["predicted_class"] == "normal"


def test_parse_with_text_around():
    text = "Voici le résultat:\n```json\n" + json.dumps(VALID) + "\n```\nMerci."
    parsed, valid = parse_json(text)
    assert valid is True


def test_parse_malformed():
    parsed, valid = parse_json("not a json at all")
    assert valid is False


def test_parse_aliases_and_normalization():
    aliased = dict(VALID)
    aliased["image_quality"] = "poor"
    aliased["visual_evidence"] = ["opacité possible"]
    del aliased["visual_findings"]
    aliased["limitations"] = "Vue unique frontale."
    parsed, valid = parse_json(json.dumps(aliased))
    assert valid is True
    assert parsed["image_quality"] == "insufficient"
    assert parsed["visual_findings"] == ["opacité possible"]
    assert parsed["limitations"] == ["Vue unique frontale."]


def test_uncertainty_invalid():
    final, reason = apply_uncertainty_rule(None, False, "ok")
    assert final == "incertain"
    assert reason == "json_invalid"


def test_uncertainty_low_conf():
    p = dict(VALID)
    p["confidence"] = 0.4
    final, reason = apply_uncertainty_rule(p, True, "ok")
    assert final == "incertain"
    assert "low_confidence" in reason


def test_uncertainty_image_insufficient():
    final, reason = apply_uncertainty_rule(VALID, True, "insufficient")
    assert final == "incertain"


def test_uncertainty_ensemble_disagree():
    final, reason = apply_uncertainty_rule(
        VALID, True, "ok", ensemble_classes=["normal", "suspicion_opacite", "normal"]
    )
    assert final == "incertain"
    assert reason == "ensemble_disagreement"


def test_uncertainty_ensemble_agree():
    final, reason = apply_uncertainty_rule(
        VALID, True, "ok", ensemble_classes=["normal", "normal", "normal"]
    )
    assert final == "normal"
