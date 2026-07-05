# Assistant Radiologue Virtuel - Prototype pédagogique

> Ce dépôt démontre une chaîne d'ingénierie en IA multimodale sur radiographies thoraciques frontales. Il ne constitue pas un dispositif médical.

## But du projet

Application web Gradio qui analyse une radiographie thoracique frontale et retourne un JSON structuré avec trois classes possibles :

- `normal`
- `suspicion_opacite`
- `incertain`

Le projet compare plusieurs configurations de prompting et de modèle, journalise les runs dans SQLite, affiche un dashboard de métriques et exporte un rapport de soutenance.

## Schéma JSON attendu

Le schéma interne attendu est :

```json
{
  "image_quality": "ok | degraded | insufficient",
  "predicted_class": "normal | suspicion_opacite | incertain",
  "confidence": 0.0,
  "visual_findings": ["observation courte"],
  "justification": "justification synthétique et prudente",
  "limitations": ["limite identifiée"],
  "warning": "Prototype pédagogique. Non destiné au diagnostic. Validation par un professionnel qualifié requise."
}
```

Le parseur accepte aussi les alias documentaires suivants :

- `visual_evidence` -> `visual_findings`
- `good | limited | poor` -> `ok | degraded | insufficient`

## Stack

- Python 3.11
- PyTorch
- Hugging Face Transformers
- bitsandbytes
- Pillow
- pydicom
- pandas
- matplotlib
- Gradio
- SQLite
- pytest

