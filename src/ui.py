"""UI Gradio : analyse, dashboard, registre d'erreurs et mode démo."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .config import CONFIGS, DB_PATH, WARNING_TEXT
from .db import (
    fetch_errors,
    fetch_models,
    fetch_runs,
    init_db,
    insert_error,
    insert_run,
    register_model,
    register_prompt,
)
from .inference import run_ensemble, run_single
from .metrics import compute_metrics, confusion_matrix
from .models import free_model, load_model
from .postprocess import apply_uncertainty_rule, auto_error_type, parse_json
from .preprocessing import preprocess
from .prompts import BASELINE, ENSEMBLE

WARNING_BANNER = (
    "<div style='background:#b00020;color:white;padding:14px;border-radius:6px;"
    "font-weight:600;font-size:15px;margin-bottom:12px;text-align:center;'>"
    f"⚠️ {WARNING_TEXT}</div>"
)

_loaded: Dict[str, Any] = {"hf_id": None, "model": None, "processor": None}
ERROR_TABLE_COLUMNS = [
    "run_id",
    "case_id",
    "model",
    "ground_truth",
    "pred_class",
    "confidence",
    "json_valid",
    "auto_error",
    "reason",
    "error_message",
]
ERROR_LOG_COLUMNS = [
    "error_id",
    "run_id",
    "case_id",
    "model",
    "error_type",
    "comment",
    "created_at",
]


def _ensure_model(hf_id: str):
    if _loaded["hf_id"] == hf_id and _loaded["model"] is not None:
        return _loaded["model"], _loaded["processor"]
    if _loaded["model"] is not None:
        free_model(_loaded["model"], _loaded["processor"])
        _loaded["model"] = None
        _loaded["processor"] = None
    model, processor = load_model(hf_id)
    _loaded["hf_id"] = hf_id
    _loaded["model"] = model
    _loaded["processor"] = processor
    return model, processor


def analyse(image_file, config_name: str, ground_truth: str = "unknown"):
    if image_file is None:
        return None, {}, "*Aucune image chargée*", None, ""

    path = image_file if isinstance(image_file, str) else getattr(image_file, "name", None)
    if path is None:
        return None, {}, "*Erreur fichier*", None, ""

    image, qc = preprocess(path)
    cfg = CONFIGS[config_name]
    model_id = register_model(name=config_name, hf_id=cfg["hf_id"])
    try:
        model, processor = _ensure_model(cfg["hf_id"])
    except Exception as exc:
        prompt_name = "baseline" if cfg["prompt_kind"] == "baseline" else "ensemble_reinforced"
        prompt_text = BASELINE if cfg["prompt_kind"] == "baseline" else "\n--SEP--\n".join(text for _, text in ENSEMBLE)
        prompt_id = register_prompt(prompt_name, "v1", cfg["prompt_kind"], prompt_text)
        true_label = ground_truth if ground_truth and ground_truth != "unknown" else None
        fallback_json = {
            "image_quality": qc["image_quality"],
            "predicted_class": "incertain",
            "confidence": 0.0,
            "visual_findings": [],
            "justification": "Analyse live indisponible dans cet environnement.",
            "limitations": [f"Chargement du modèle impossible: {exc}"],
            "warning": WARNING_TEXT,
        }
        insert_run(
            case_id=Path(path).stem,
            image_path=str(path),
            true_label=true_label,
            model_id=model_id,
            prompt_id=prompt_id,
            prompt_kind=cfg["prompt_kind"],
            raw_output="",
            parsed_json=fallback_json,
            pred_class="incertain",
            confidence=0.0,
            ground_truth=true_label,
            latency_ms=0,
            json_valid=False,
            reason="model_unavailable_ui",
            warning_text=WARNING_TEXT,
            error_message=str(exc),
        )
        class_md = (
            "### Classe finale : "
            "<span style='background:#6b7280;color:white;padding:4px 10px;border-radius:4px;'>incertain</span>  \n"
            "**Raison** : `model_unavailable_ui`  \n"
            "**Confiance modèle** : 0.0  \n"
            f"**QC image** : `{qc['image_quality']}` (mean={qc['luminance_mean']}, std={qc['luminance_std']})  \n"
            "**Latence** : 0 ms"
        )
        return image, fallback_json, class_md, 0.0, "Analyse live indisponible dans cet environnement."

    if cfg["prompt_kind"] == "baseline":
        prompt_id = register_prompt("baseline", "v1", "single", BASELINE)
        raw_output, latency_ms = run_single(model, processor, image, BASELINE)
        parsed, valid = parse_json(raw_output)
        final_class, reason = apply_uncertainty_rule(parsed, valid, qc["image_quality"])
        confidence = float(parsed.get("confidence")) if parsed and "confidence" in parsed else None
        display_json = parsed if parsed is not None else {"raw_output": raw_output, "warning": WARNING_TEXT}
    else:
        prompt_id = register_prompt(
            "ensemble_reinforced",
            "v1",
            "ensemble",
            "\n--SEP--\n".join(text for _, text in ENSEMBLE),
        )
        ensemble_results = run_ensemble(model, processor, image, ENSEMBLE)
        votes = []
        parsed_each = []
        total_latency = 0
        for prompt_name, raw_text, latency in ensemble_results:
            parsed_i, valid_i = parse_json(raw_text)
            votes.append(parsed_i["predicted_class"] if valid_i and parsed_i else "incertain")
            parsed_each.append(
                {"prompt": prompt_name, "parsed": parsed_i, "valid": valid_i, "latency_ms": latency}
            )
            total_latency += latency
        parsed = next((item["parsed"] for item in parsed_each if item["valid"]), None)
        valid = parsed is not None
        final_class, reason = apply_uncertainty_rule(parsed, valid, qc["image_quality"], ensemble_classes=votes)
        confidence = float(parsed.get("confidence")) if parsed and "confidence" in parsed else None
        latency_ms = total_latency
        raw_output = json.dumps(parsed_each, ensure_ascii=False, default=str)
        display_json = {
            "ensemble_votes": votes,
            "primary": parsed,
            "final_class": final_class,
            "reason": reason,
            "warning": WARNING_TEXT,
        }

    true_label = ground_truth if ground_truth and ground_truth != "unknown" else None
    insert_run(
        case_id=Path(path).stem,
        image_path=str(path),
        true_label=true_label,
        model_id=model_id,
        prompt_id=prompt_id,
        prompt_kind=cfg["prompt_kind"],
        raw_output=raw_output,
        parsed_json=parsed,
        pred_class=final_class,
        confidence=confidence,
        ground_truth=true_label,
        latency_ms=latency_ms,
        json_valid=valid,
        reason=reason,
        warning_text=(parsed or {}).get("warning", WARNING_TEXT),
    )

    badge_color = {
        "normal": "#1b8e3a",
        "suspicion_opacite": "#d97706",
        "incertain": "#6b7280",
    }.get(final_class, "#6b7280")
    class_md = (
        "### Classe finale : "
        f"<span style='background:{badge_color};color:white;padding:4px 10px;border-radius:4px;'>{final_class}</span>  \n"
        f"**Raison** : `{reason}`  \n"
        f"**Confiance modèle** : {confidence if confidence is not None else '—'}  \n"
        f"**QC image** : `{qc['image_quality']}` (mean={qc['luminance_mean']}, std={qc['luminance_std']})  \n"
        f"**Latence** : {latency_ms} ms"
    )
    justification = parsed.get("justification", "") if parsed else ""
    return image, display_json, class_md, confidence, justification


def _runs_to_dicts(rows) -> List[Dict[str, Any]]:
    return [dict(row) for row in rows]


def _empty_dataframe(columns: List[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def _empty_plot(message: str):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.text(0.5, 0.5, message, ha="center", va="center")
    ax.set_axis_off()
    return fig


def _plot_confusion_matrices(matrices):
    count = len(matrices)
    fig, axes = plt.subplots(1, count, figsize=(5 * count, 4.5), squeeze=False)
    for index, (name, matrix, classes) in enumerate(matrices):
        ax = axes[0, index]
        image = ax.imshow(matrix, cmap="Blues")
        ax.set_title(name, fontsize=10)
        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=30, ha="right", fontsize=8)
        ax.set_yticklabels(classes, fontsize=8)
        ax.set_xlabel("Prédiction", fontsize=8)
        ax.set_ylabel("Vérité terrain", fontsize=8)
        for row_idx in range(matrix.shape[0]):
            for col_idx in range(matrix.shape[1]):
                color = "white" if matrix[row_idx, col_idx] > matrix.max() / 2 else "black"
                ax.text(col_idx, row_idx, str(matrix[row_idx, col_idx]), ha="center", va="center", color=color, fontsize=9)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def _plot_latency_bars(df: pd.DataFrame):
    if df.empty or "latency_mean_ms" not in df:
        return _empty_plot("Pas de latence disponible")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(df["model"], df["latency_mean_ms"], color=["#4f46e5", "#0891b2", "#d97706"][: len(df)])
    ax.set_ylabel("Latence moyenne (ms)")
    ax.set_title("Comparaison des latences")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    return fig


def compute_dashboard():
    init_db()
    models = fetch_models()
    if not models:
        empty = pd.DataFrame([{"info": "Aucune donnée. Lance scripts/run_eval.py ou utilise un runs.sqlite pré-calculé."}])
        return empty, _empty_plot("Pas de runs"), _empty_plot("Pas de latence")

    rows = []
    matrices = []
    for model in models:
        runs = _runs_to_dicts(fetch_runs(model_id=model["id"]))
        gt_runs = [run for run in runs if run.get("ground_truth")]
        metrics = compute_metrics(gt_runs) if gt_runs else {"n": 0}
        metrics["model"] = model["name"]
        rows.append(metrics)
        if gt_runs:
            matrix, classes = confusion_matrix(gt_runs)
            matrices.append((model["name"], matrix, classes))

    df = pd.DataFrame(rows)
    confusion_fig = _plot_confusion_matrices(matrices) if matrices else _empty_plot("Aucune ground truth")
    latency_fig = _plot_latency_bars(df)
    return df, confusion_fig, latency_fig


def fetch_errors_table() -> pd.DataFrame:
    init_db()
    runs = _runs_to_dicts(fetch_runs())
    rows = []
    for run in runs:
        auto_error = auto_error_type(run["pred_class"], run.get("ground_truth"), bool(run["json_valid"]))
        if run.get("error_message"):
            auto_error = "erreur_technique"
        elif run.get("reason") == "image_insufficient":
            auto_error = "image_mauvaise_qualite"
        rows.append(
            {
                "run_id": run["id"],
                "case_id": run.get("case_id"),
                "model": run.get("model_name"),
                "ground_truth": run.get("ground_truth"),
                "pred_class": run["pred_class"],
                "confidence": run.get("confidence"),
                "json_valid": bool(run["json_valid"]),
                "auto_error": auto_error or "",
                "reason": run.get("reason", ""),
                "error_message": run.get("error_message", ""),
            }
        )
    if not rows:
        return _empty_dataframe(ERROR_TABLE_COLUMNS)
    return pd.DataFrame(rows, columns=ERROR_TABLE_COLUMNS)


def add_error_comment(run_id: int, error_type: str, comment: str) -> str:
    if not run_id:
        return "Renseigne un run_id."
    insert_error(int(run_id), error_type or "manual", comment or "")
    return f"Commentaire ajouté pour run {run_id}."


def fetch_errors_df() -> pd.DataFrame:
    rows = [dict(row) for row in fetch_errors()]
    if not rows:
        return _empty_dataframe(ERROR_LOG_COLUMNS)
    return pd.DataFrame(rows, columns=ERROR_LOG_COLUMNS)


def _latest_case_choices() -> list[str]:
    runs = _runs_to_dicts(fetch_runs())
    seen = []
    for run in reversed(runs):
        key = f"{run.get('case_id')} | {run.get('model_name')}"
        if run.get("case_id") and key not in seen:
            seen.append(key)
    return seen or ["Aucun run disponible"]


def load_saved_case(choice: str):
    if not choice or choice == "Aucun run disponible":
        return None, {}, "Aucun run sauvegardé."
    case_id, model_name = [part.strip() for part in choice.split("|", 1)]
    runs = _runs_to_dicts(fetch_runs())
    run = next(
        (item for item in reversed(runs) if item.get("case_id") == case_id and item.get("model_name") == model_name),
        None,
    )
    if run is None:
        return None, {}, "Run introuvable."
    image_path = Path(run["image_path"])
    if not image_path.is_absolute():
        image_path = (Path(__file__).resolve().parents[1] / image_path).resolve()
    parsed = run.get("parsed_json")
    if isinstance(parsed, str):
        try:
            parsed = json.loads(parsed)
        except Exception:
            parsed = {"raw_parsed_json": parsed}
    summary = (
        f"**case_id**: `{run.get('case_id')}`  \n"
        f"**modèle**: `{run.get('model_name')}`  \n"
        f"**vérité terrain**: `{run.get('ground_truth')}`  \n"
        f"**prédiction**: `{run.get('pred_class')}`  \n"
        f"**latence**: `{run.get('latency_ms')}` ms  \n"
        f"**json_valid**: `{bool(run.get('json_valid'))}`  \n"
        f"**raison**: `{run.get('reason', '')}`"
    )
    return str(image_path), parsed or {}, summary


def refresh_saved_case_choices():
    choices = _latest_case_choices()
    return gr.Dropdown(choices=choices, value=choices[0] if choices else None)


def build_ui() -> gr.Blocks:
    init_db(DB_PATH)
    with gr.Blocks(title="Assistant Radiologue Virtuel - Prototype pédagogique") as demo:
        gr.HTML(WARNING_BANNER)
        gr.Markdown(
            "# Assistant Radiologue Virtuel\n"
            "*Prototype pédagogique non clinique. Les résultats sauvegardés peuvent être utilisés en mode démo sans inférence live.*"
        )

        with gr.Tabs():
            with gr.Tab("Analyse"):
                with gr.Row():
                    with gr.Column(scale=1):
                        img_in = gr.File(
                            label="Radiographie thoracique frontale (PNG/JPG/DICOM)",
                            file_types=[".png", ".jpg", ".jpeg", ".dcm"],
                        )
                        cfg_radio = gr.Radio(
                            choices=list(CONFIGS.keys()),
                            value="medgemma_baseline",
                            label="Configuration",
                            info="\n".join(f"- {name}: {cfg['label']}" for name, cfg in CONFIGS.items()),
                        )
                        gt_radio = gr.Radio(
                            choices=["unknown", "normal", "suspicion_opacite"],
                            value="unknown",
                            label="Ground truth (optionnel)",
                        )
                        analyse_btn = gr.Button("Analyser", variant="primary")
                    with gr.Column(scale=1):
                        img_out = gr.Image(label="Image préprocessée (RGB 896)")
                        class_md = gr.Markdown()
                        conf_num = gr.Number(label="Confiance modèle", precision=3)
                        justif_box = gr.Textbox(label="Justification", lines=3)
                json_out = gr.JSON(label="Sortie JSON structurée")

                analyse_btn.click(
                    analyse,
                    inputs=[img_in, cfg_radio, gt_radio],
                    outputs=[img_out, json_out, class_md, conf_num, justif_box],
                )

            with gr.Tab("Dashboard"):
                refresh_btn = gr.Button("Recharger métriques")
                metrics_df = gr.Dataframe(label="Métriques par configuration", interactive=False)
                cm_plot = gr.Plot(label="Matrices de confusion")
                latency_plot = gr.Plot(label="Latence moyenne")
                refresh_btn.click(compute_dashboard, outputs=[metrics_df, cm_plot, latency_plot])
                demo.load(compute_dashboard, outputs=[metrics_df, cm_plot, latency_plot])

            with gr.Tab("Démo sauvegardée"):
                gr.Markdown(
                    "Ce mode relit des résultats déjà stockés dans `runs.sqlite`. "
                    "Il sert de secours si le modèle, CUDA ou le token HF ne sont pas disponibles."
                )
                saved_choice = gr.Dropdown(choices=_latest_case_choices(), label="Cas sauvegardé")
                saved_refresh = gr.Button("Recharger la liste")
                saved_image = gr.Image(label="Image sauvegardée")
                saved_json = gr.JSON(label="JSON sauvegardé")
                saved_md = gr.Markdown()
                saved_choice.change(load_saved_case, inputs=[saved_choice], outputs=[saved_image, saved_json, saved_md])
                saved_refresh.click(refresh_saved_case_choices, outputs=[saved_choice])

            with gr.Tab("Registre d'erreurs"):
                gr.Markdown(
                    "Auto-tags disponibles : faux positif, faux négatif, incertain à analyser, JSON invalide, "
                    "erreur technique, image de mauvaise qualité."
                )
                err_refresh = gr.Button("Rafraîchir")
                err_df = gr.Dataframe(label="Synthèse des erreurs", interactive=False)
                with gr.Row():
                    rid = gr.Number(label="run_id", precision=0)
                    etype = gr.Textbox(label="error_type", value="hallucination", scale=1)
                    comment = gr.Textbox(label="commentaire", scale=2)
                    add_btn = gr.Button("Ajouter")
                add_status = gr.Markdown()
                err_log = gr.Dataframe(label="Commentaires enregistrés", interactive=False)

                err_refresh.click(fetch_errors_table, outputs=err_df)
                err_refresh.click(fetch_errors_df, outputs=err_log)
                add_btn.click(add_error_comment, inputs=[rid, etype, comment], outputs=add_status).then(
                    fetch_errors_df, outputs=err_log
                )
                demo.load(fetch_errors_table, outputs=err_df)
                demo.load(fetch_errors_df, outputs=err_log)

            with gr.Tab("À propos"):
                gr.Markdown(
                    f"""
**Configurations comparées**
- `gemma3_baseline` : Gemma-3 4B, prompt baseline
- `medgemma_baseline` : MedGemma 4B, prompt baseline
- `medgemma_ensemble_reinforced` : MedGemma 4B, ensemble de 3 prompts renforcés

**Règle d'incertitude**
- JSON invalide -> `incertain`
- image_quality == insufficient -> `incertain`
- confidence < 0.60 -> `incertain`
- désaccord d'ensemble -> `incertain`

**Schéma JSON harmonisé**
- sortie interne : `visual_findings`, `image_quality` = `ok | degraded | insufficient`
- alias acceptés au parsing : `visual_evidence`, `good | limited | poor`

**Mode secours**
- `python scripts/run_eval.py --offline` : relit seulement les runs existants
- onglet *Démo sauvegardée* : réaffiche image + JSON sans relancer le modèle

**Avertissement**
- {WARNING_TEXT}
"""
                )

        gr.HTML(WARNING_BANNER)
    return demo
