"""Post-traitement : parse JSON robuste + règle d'incertitude combinée."""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from .config import CONF_THRESHOLD, WARNING_TEXT

REQUIRED_KEYS = (
    "image_quality",
    "predicted_class",
    "confidence",
    "visual_findings",
    "justification",
    "limitations",
    "warning",
)

VALID_CLASSES = {"normal", "suspicion_opacite", "incertain"}
VALID_QUALITY = {"ok", "degraded", "insufficient"}
QUALITY_ALIASES = {
    "good": "ok",
    "limited": "degraded",
    "poor": "insufficient",
}

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _normalize_schema(parsed: Dict[str, Any]) -> Dict[str, Any]:
    """Accepte quelques alias documentaires et les ramène au schéma interne."""
    normalized = dict(parsed)

    if "visual_findings" not in normalized and "visual_evidence" in normalized:
        normalized["visual_findings"] = normalized["visual_evidence"]

    quality = normalized.get("image_quality")
    if isinstance(quality, str):
        normalized["image_quality"] = QUALITY_ALIASES.get(quality, quality)

    limitations = normalized.get("limitations")
    if isinstance(limitations, str):
        normalized["limitations"] = [limitations]

    return normalized


def parse_json(raw_text: str) -> Tuple[Optional[Dict[str, Any]], bool]:
    """Extrait le premier bloc JSON-like et le valide."""
    if not raw_text:
        return None, False

    text = raw_text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    candidates = []
    try:
        candidates.append(json.loads(text))
    except Exception:
        pass

    match = _JSON_RE.search(raw_text)
    if match:
        try:
            candidates.append(json.loads(match.group(0)))
        except Exception:
            pass

    parsed = next((candidate for candidate in candidates if isinstance(candidate, dict)), None)
    if parsed is None:
        return None, False
    parsed = _normalize_schema(parsed)

    if not all(key in parsed for key in REQUIRED_KEYS):
        return parsed, False
    if parsed.get("predicted_class") not in VALID_CLASSES:
        return parsed, False
    if parsed.get("image_quality") not in VALID_QUALITY:
        return parsed, False

    try:
        confidence = float(parsed.get("confidence", 0.0))
    except (TypeError, ValueError):
        return parsed, False
    if not (0.0 <= confidence <= 1.0):
        return parsed, False
    if not isinstance(parsed.get("visual_findings"), list):
        return parsed, False
    if not isinstance(parsed.get("limitations"), list):
        return parsed, False

    parsed["confidence"] = confidence
    parsed["warning"] = WARNING_TEXT
    return parsed, True


def apply_uncertainty_rule(
    parsed: Optional[Dict[str, Any]],
    json_valid: bool,
    image_quality: Optional[str] = None,
    ensemble_classes: Optional[List[str]] = None,
    threshold: float = CONF_THRESHOLD,
) -> Tuple[str, str]:
    """Applique la règle d'incertitude combinée."""
    if not json_valid or parsed is None:
        return "incertain", "json_invalid"

    qc = image_quality or parsed.get("image_quality")
    if qc == "insufficient":
        return "incertain", "image_insufficient"

    confidence = float(parsed.get("confidence", 0.0))
    if confidence < threshold:
        return "incertain", f"low_confidence<{threshold}"

    if ensemble_classes:
        majority = max(set(ensemble_classes), key=ensemble_classes.count)
        if any(pred != majority for pred in ensemble_classes):
            return "incertain", "ensemble_disagreement"
        return majority, "ensemble_agreement"

    return parsed["predicted_class"], "ok"


def auto_error_type(pred: str, ground_truth: Optional[str], json_valid: bool) -> Optional[str]:
    """Classifie automatiquement le type d'erreur d'un run."""
    if not json_valid:
        return "json_invalide"
    if ground_truth is None:
        return None
    if pred == "incertain":
        return "incertain_a_analyser"
    if pred == "suspicion_opacite" and ground_truth == "normal":
        return "faux_positif"
    if pred == "normal" and ground_truth == "suspicion_opacite":
        return "faux_negatif"
    if pred == ground_truth:
        return None
    return "autre"
