"""Genere les rapports et infographies a partir de runs.sqlite."""
from __future__ import annotations

import csv
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib"))

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import DB_PATH, LABELS_CSV, REPORT_DIR, WARNING_TEXT  # noqa: E402
from src.db import fetch_errors, fetch_runs, init_db  # noqa: E402
from src.metrics import compute_metrics, confusion_matrix, summarize_errors  # noqa: E402
from src.postprocess import auto_error_type  # noqa: E402

TARGET_COMMENTED_CASES = 24
PRESENTATION_LABELS = {
    "gemma3_baseline": "Gemma 3 baseline",
    "medgemma_baseline": "MedGemma baseline",
    "medgemma_ensemble_reinforced": "MedGemma renforce",
}


def _parse_payload(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
            return decoded if isinstance(decoded, dict) else {}
        except Exception:
            return {}
    return {}


def _effective_payload(value: Any) -> Dict[str, Any]:
    """Retourne le payload clinique principal, y compris pour les ensembles."""
    payload = _parse_payload(value)
    primary = payload.get("primary")
    if isinstance(primary, dict):
        merged = dict(primary)
        if "ensemble_votes" in payload:
            merged["ensemble_votes"] = payload["ensemble_votes"]
        if "qc" in payload:
            merged["qc"] = payload["qc"]
        return merged
    return payload


def _model_label(model_name: str) -> str:
    return PRESENTATION_LABELS.get(model_name, model_name)


def _run_bucket(run: Dict[str, Any]) -> str:
    gt = run.get("ground_truth")
    pred = run.get("pred_class")
    json_valid = bool(run.get("json_valid"))
    if run.get("error_message"):
        return "technical_errors"
    if not json_valid:
        return "json_invalid"
    if pred == "incertain":
        return "uncertain"
    if gt and pred == gt and pred != "incertain":
        return "success"
    if pred == "suspicion_opacite" and gt == "normal":
        return "false_positive"
    if pred == "normal" and gt == "suspicion_opacite":
        return "false_negative"
    return "other"


def select_commented_cases(rows: Iterable[Dict[str, Any]], target_count: int = TARGET_COMMENTED_CASES) -> List[Dict[str, Any]]:
    buckets = {
        "false_negative": [],
        "false_positive": [],
        "uncertain": [],
        "json_invalid": [],
        "technical_errors": [],
        "success": [],
        "other": [],
    }
    seen = set()
    ordered = sorted(
        rows,
        key=lambda run: (
            str(run.get("case_id", "")),
            str(run.get("model_name", "")),
            int(run.get("id", 0)),
        ),
    )
    for run in ordered:
        key = (str(run.get("case_id")), str(run.get("model_name")))
        if key in seen:
            continue
        seen.add(key)
        buckets[_run_bucket(run)].append(run)

    selected: List[Dict[str, Any]] = []
    quotas = [
        ("false_negative", 4),
        ("false_positive", 4),
        ("uncertain", 4),
        ("json_invalid", 2),
        ("technical_errors", 2),
        ("success", 8),
    ]
    for name, wanted in quotas:
        selected.extend(buckets[name][:wanted])
        if len(selected) >= target_count:
            return selected[:target_count]

    for name in ("other", "success", "uncertain", "false_positive", "false_negative"):
        for run in buckets[name]:
            if run not in selected:
                selected.append(run)
            if len(selected) >= target_count:
                return selected[:target_count]
    return selected[:target_count]


def _comment_line(run: Dict[str, Any]) -> str:
    parsed = _effective_payload(run.get("parsed_json"))
    findings = parsed.get("visual_findings") or parsed.get("visual_evidence") or []
    limitations = parsed.get("limitations") or []
    if isinstance(limitations, str):
        limitations = [limitations]
    if isinstance(findings, str):
        findings = [findings]
    return (
        f"- case=`{run.get('case_id')}` modele=`{run.get('model_name')}` "
        f"gt=`{run.get('ground_truth')}` pred=`{run.get('pred_class')}` "
        f"raison=`{run.get('reason', '')}` json_valid=`{bool(run.get('json_valid'))}` "
        f"findings=`{findings}` limitations=`{limitations[:2]}`"
    )


def _unique_label_counts(rows: Iterable[Dict[str, Any]]) -> Counter:
    labels_by_case: Dict[str, str] = {}
    for run in rows:
        case_id = run.get("case_id")
        ground_truth = run.get("ground_truth")
        if case_id and ground_truth in {"normal", "suspicion_opacite"}:
            labels_by_case[str(case_id)] = str(ground_truth)
    return Counter(labels_by_case.values())


def _metric_table(by_model_runs: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    return {model_name: compute_metrics(runs_m) for model_name, runs_m in by_model_runs.items()}


def _save_metric_comparison(metrics_by_model: Dict[str, Dict[str, Any]], out_path: Path) -> None:
    names = list(metrics_by_model)
    labels = [_model_label(name) for name in names]
    metric_keys = ["accuracy", "macro_f1", "sensitivity", "specificity"]
    metric_labels = ["Accuracy", "Macro-F1", "Sensibilite", "Specificite"]
    x = np.arange(len(metric_keys))
    width = 0.8 / max(len(names), 1)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = ["#355C7D", "#F67280", "#6C5B7B", "#99B898"]
    for idx, name in enumerate(names):
        values = [metrics_by_model[name].get(key, 0) for key in metric_keys]
        offset = (idx - (len(names) - 1) / 2) * width
        bars = ax.bar(x + offset, values, width, label=labels[idx], color=colors[idx % len(colors)])
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)

    ax.set_title("Comparaison des performances de classification")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.08)
    ax.set_xticks(x, metric_labels)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=max(1, len(names)))
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _save_error_breakdown(metrics_by_model: Dict[str, Dict[str, Any]], out_path: Path) -> None:
    names = list(metrics_by_model)
    labels = [_model_label(name) for name in names]
    keys = ["false_positives", "false_negatives", "uncertain_cases", "json_invalid_cases"]
    key_labels = ["Faux positifs", "Faux negatifs", "Incertains", "JSON invalides"]
    data = np.array([[metrics_by_model[name].get(key, 0) for key in keys] for name in names])

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bottom = np.zeros(len(names))
    colors = ["#E76F51", "#B23A48", "#E9C46A", "#8D99AE"]
    for idx, key_label in enumerate(key_labels):
        values = data[:, idx] if len(data) else []
        bars = ax.bar(labels, values, bottom=bottom, label=key_label, color=colors[idx])
        ax.bar_label(bars, labels=[str(int(v)) if v else "" for v in values], label_type="center", fontsize=8)
        bottom += values

    ax.set_title("Typologie des erreurs et garde-fous")
    ax.set_ylabel("Nombre de cas")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _save_latency_json(metrics_by_model: Dict[str, Dict[str, Any]], out_path: Path) -> None:
    names = list(metrics_by_model)
    labels = [_model_label(name) for name in names]
    latency_s = [metrics_by_model[name].get("latency_mean_ms", 0) / 1000 for name in names]
    json_valid = [metrics_by_model[name].get("json_valid_pct", 0) for name in names]

    fig, ax1 = plt.subplots(figsize=(10, 5.5))
    bars = ax1.bar(labels, latency_s, color="#2A9D8F", alpha=0.85)
    ax1.bar_label(bars, fmt="%.1fs", padding=3, fontsize=8)
    ax1.set_ylabel("Latence moyenne (secondes)")
    ax1.set_title("Compromis latence / validite JSON")
    ax1.grid(axis="y", alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(labels, json_valid, color="#264653", marker="o", linewidth=2.5, label="JSON valide")
    ax2.set_ylabel("JSON valide (%)")
    ax2.set_ylim(0, 110)
    for idx, value in enumerate(json_valid):
        ax2.annotate(f"{value:.0f}%", (idx, value), textcoords="offset points", xytext=(0, 8), ha="center")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _save_confusion_matrices(by_model_runs: Dict[str, List[Dict[str, Any]]], out_dir: Path) -> List[Path]:
    paths: List[Path] = []
    for model_name, runs_m in by_model_runs.items():
        matrix, classes = confusion_matrix(runs_m)
        out_path = out_dir / f"confusion_{model_name}.png"
        fig, ax = plt.subplots(figsize=(6, 5.5))
        image = ax.imshow(matrix, cmap="Blues")
        ax.set_title(f"Matrice de confusion - {_model_label(model_name)}")
        ax.set_xlabel("Prediction")
        ax.set_ylabel("Verite terrain")
        ax.set_xticks(np.arange(len(classes)), classes, rotation=25, ha="right")
        ax.set_yticks(np.arange(len(classes)), classes)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="#111")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
        paths.append(out_path)
    return paths


def _write_qualitative_report(
    *,
    path: Path,
    rows: List[Dict[str, Any]],
    by_case: Dict[str, List[Dict[str, Any]]],
    model_names: List[str],
    metrics_by_model: Dict[str, Dict[str, Any]],
    label_counts: Counter,
    infographic_paths: Dict[str, Path],
    confusion_paths: List[Path],
) -> None:
    baseline = metrics_by_model.get("gemma3_baseline", {})
    reinforced = metrics_by_model.get("medgemma_ensemble_reinforced", {})
    macro_gain = None
    fp_delta = None
    if baseline and reinforced:
        macro_gain = round(reinforced.get("macro_f1", 0) - baseline.get("macro_f1", 0), 4)
        fp_delta = int(baseline.get("false_positives", 0) - reinforced.get("false_positives", 0))

    path.write_text(
        "\n".join(
            [
                "# Rapport qualitatif - Assistant Radiologue Virtuel",
                "",
                f"> **{WARNING_TEXT}**",
                "",
                "## Positionnement",
                "",
                "Le projet repond a un objectif pedagogique : construire une chaine IA medicale prudente, "
                "mesurable et tracable pour des radiographies thoraciques frontales. Le systeme ne pretend "
                "pas diagnostiquer une pneumonie ; il classe les images en `normal`, `suspicion_opacite` ou "
                "`incertain`, puis documente ses limites.",
                "",
                "## Couverture du cahier des charges",
                "",
                "- Le socle Must have est couvert : application web, upload d'image, sortie JSON, warning, logs SQLite et baseline reproductible.",
                "- Le niveau Should have est largement couvert : comparaison baseline vs version renforcee, metriques, matrices de confusion, latence, validite JSON et analyse d'erreurs.",
                "- Le niveau Could have a ete explore via un essai LoRA/QLoRA sur 250 cas, mais non retenu dans le resultat final car les premieres sorties etaient instables au format JSON.",
                "",
                "## Donnees et protocole",
                "",
                f"- Dataset : RSNA Pneumonia Detection Challenge.",
                f"- Cas uniques evalues : {len(by_case)}.",
                f"- Repartition reelle : normal={label_counts.get('normal', 0)}, suspicion_opacite={label_counts.get('suspicion_opacite', 0)}.",
                f"- Configurations comparees : {', '.join(f'`{name}`' for name in model_names)}.",
                "",
                "Le meme jeu de 30 cas est utilise pour comparer les configurations. Cette taille reste limitee, "
                "mais elle est suffisante pour demontrer le protocole d'evaluation, les garde-fous et les limites.",
                "",
                "## Resultats quantitatifs",
                "",
                f"![Comparaison des performances]({infographic_paths['metrics'].name})",
                "",
                f"![Typologie des erreurs]({infographic_paths['errors'].name})",
                "",
                f"![Latence et JSON]({infographic_paths['latency'].name})",
                "",
                *(f"![Matrice de confusion - {confusion_path.stem.replace('confusion_', '')}]({confusion_path.name})" for confusion_path in confusion_paths),
                "",
                "Synthese des resultats :",
                "",
                f"- Gemma baseline obtient accuracy={baseline.get('accuracy', 'n/a')}, macro-F1={baseline.get('macro_f1', 'n/a')}, specificite={baseline.get('specificity', 'n/a')} et {baseline.get('false_positives', 'n/a')} faux positifs.",
                f"- MedGemma renforce obtient accuracy={reinforced.get('accuracy', 'n/a')}, macro-F1={reinforced.get('macro_f1', 'n/a')}, specificite={reinforced.get('specificity', 'n/a')} et {reinforced.get('false_positives', 'n/a')} faux positifs.",
                f"- Gain macro-F1 observe : {macro_gain if macro_gain is not None else 'n/a'}. Reduction des faux positifs : {fp_delta if fp_delta is not None else 'n/a'}.",
                f"- Le compromis principal est la latence : MedGemma renforce est plus lent, mais plus prudent grace aux sorties `incertain`.",
                "",
                "## Lecture qualitative",
                "",
                "La baseline Gemma detecte beaucoup de cas positifs, ce qui donne une sensibilite elevee, "
                "mais elle surclasse trop souvent des images normales en suspicion d'opacite. Cela se voit dans "
                "le nombre eleve de faux positifs et dans une specificite faible.",
                "",
                "La version MedGemma renforcee corrige partiellement ce comportement : elle reduit les faux positifs, "
                "augmente le macro-F1 et introduit des decisions `incertain` lorsque les prompts ne convergent pas. "
                "Cette incertitude n'est pas un echec dans ce contexte ; elle materialise le garde-fou demande par le sujet.",
                "",
                "## Points forts",
                "",
                "- Chaine de bout en bout fonctionnelle : preparation RSNA, inference, post-traitement, SQLite, dashboard et export.",
                "- Sortie structuree et verifiable, avec validite JSON mesuree.",
                "- Comparaison experimentale defendable entre baseline et amelioration.",
                "- Documentation des limites medicales et usage non clinique explicite.",
                "- Resultats conserves dans SQLite, donc la demo reste possible sans relancer les modeles.",
                "",
                "## Limites assumees",
                "",
                "- L'evaluation porte sur 30 cas, ce qui reste un echantillon pedagogique.",
                "- Les labels RSNA indiquent une suspicion/opacite liee a la pneumonie, pas un diagnostic radiologique complet.",
                "- Les hallucinations ne sont pas annotees manuellement ; le compteur signale seulement les hallucinations taggees.",
                "- Le fine-tuning LoRA a ete teste mais non integre, car il degradait la robustesse du format JSON.",
                "- Le systeme ne doit pas etre interprete comme un outil clinique.",
                "",
                "## Conclusion",
                "",
                "Le projet atteint l'objectif principal : il ne cherche pas a promettre une performance clinique, "
                "mais a demontrer une methode d'ingenierie responsable pour une IA medicale multimodale. "
                "La progression baseline vers MedGemma renforcee est mesurable, les erreurs sont visibles, "
                "et les limites sont suffisamment explicites pour une soutenance technique.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _latex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def _write_latex_report(
    *,
    path: Path,
    model_names: List[str],
    metrics_by_model: Dict[str, Dict[str, Any]],
    label_counts: Counter,
    by_case: Dict[str, List[Dict[str, Any]]],
    confusion_paths: List[Path],
    infographic_paths: Dict[str, Path],
) -> None:
    baseline = metrics_by_model.get("gemma3_baseline", {})
    reinforced = metrics_by_model.get("medgemma_ensemble_reinforced", {})
    macro_gain = (
        round(reinforced.get("macro_f1", 0) - baseline.get("macro_f1", 0), 4)
        if baseline and reinforced
        else "n/a"
    )
    fp_delta = (
        int(baseline.get("false_positives", 0) - reinforced.get("false_positives", 0))
        if baseline and reinforced
        else "n/a"
    )

    metric_rows = []
    for model_name in model_names:
        metrics = metrics_by_model.get(model_name, {})
        metric_rows.append(
            " & ".join(
                [
                    _latex_escape(_model_label(model_name)),
                    str(metrics.get("n", "")),
                    str(metrics.get("accuracy", "")),
                    str(metrics.get("macro_f1", "")),
                    str(metrics.get("sensitivity", "")),
                    str(metrics.get("specificity", "")),
                    str(metrics.get("uncertain_pct", "")),
                    str(metrics.get("false_positives", "")),
                    str(metrics.get("false_negatives", "")),
                    str(metrics.get("latency_mean_ms", "")),
                ]
            )
            + r" \\"
        )

    figure_blocks = [
        (infographic_paths["metrics"], "Comparaison des performances de classification."),
        (infographic_paths["errors"], "Typologie des erreurs et garde-fous."),
        (infographic_paths["latency"], "Compromis latence et validite JSON."),
    ]
    figure_blocks.extend((path_item, f"Matrice de confusion - {path_item.stem.replace('confusion_', '')}.") for path_item in confusion_paths)

    figures_tex = []
    for figure_path, caption in figure_blocks:
        figures_tex.append(
            "\n".join(
                [
                    r"\begin{figure}[H]",
                    r"\centering",
                    rf"\includegraphics[width=0.92\linewidth]{{{_latex_escape(figure_path.name)}}}",
                    rf"\caption{{{_latex_escape(caption)}}}",
                    r"\end{figure}",
                    "",
                ]
            )
        )

    tex = "\n".join(
        [
            r"\documentclass[11pt,a4paper]{article}",
            r"\usepackage[utf8]{inputenc}",
            r"\usepackage[T1]{fontenc}",
            r"\usepackage{geometry}",
            r"\usepackage{graphicx}",
            r"\usepackage{float}",
            r"\usepackage{booktabs}",
            r"\usepackage{array}",
            r"\usepackage{xcolor}",
            r"\usepackage{hyperref}",
            r"\geometry{margin=2.2cm}",
            r"\hypersetup{colorlinks=true, linkcolor=blue, urlcolor=blue}",
            r"\setlength{\parskip}{0.55em}",
            r"\setlength{\parindent}{0pt}",
            "",
            r"\begin{document}",
            "",
            r"\begin{titlepage}",
            r"\centering",
            r"{\Huge\bfseries Assistant Radiologue Virtuel\\[0.4em]}",
            r"{\Large Rapport final qualitatif et experimental\\[1.5em]}",
            r"\vfill",
            rf"\fbox{{\parbox{{0.9\linewidth}}{{\centering\textbf{{{_latex_escape(WARNING_TEXT)}}}}}}}",
            r"\vfill",
            r"{\large Prototype pedagogique d'IA medicale multimodale}",
            r"\end{titlepage}",
            "",
            r"\tableofcontents",
            r"\newpage",
            "",
            r"\section{Positionnement du projet}",
            "Le projet vise a construire un prototype pedagogique d'assistant radiologique virtuel pour radiographies thoraciques frontales. "
            "L'objectif n'est pas de produire un diagnostic medical, mais de demontrer une chaine d'ingenierie responsable : entree image, "
            "pretraitement, modele vision-langage, sortie JSON, garde-fous, journalisation, evaluation et analyse critique.",
            "",
            r"Le systeme limite volontairement ses sorties a trois classes : \texttt{normal}, \texttt{suspicion\_opacite} et \texttt{incertain}. "
            r"La classe \texttt{incertain} est traitee comme un garde-fou lorsque les indices visuels ou la confiance ne permettent pas une conclusion prudente.",
            "",
            r"\section{Couverture du cahier des charges}",
            r"\begin{itemize}",
            r"\item \textbf{Must have :} couvert. Le projet dispose d'une application web, d'une baseline, d'une sortie JSON, d'un warning obligatoire et de logs SQLite.",
            r"\item \textbf{Should have :} largement couvert. La baseline est comparee a une version renforcee avec metriques, matrices de confusion, latence, erreurs et rapport exporte.",
            r"\item \textbf{Could have :} explore. Un essai LoRA/QLoRA sur 250 cas a ete realise, mais il n'a pas ete integre au resultat final car les sorties JSON etaient instables.",
            r"\end{itemize}",
            "",
            r"\section{Donnees et protocole}",
            rf"Le jeu d'evaluation provient du RSNA Pneumonia Detection Challenge. Il contient {len(by_case)} cas uniques, "
            rf"avec une repartition reelle de {label_counts.get('normal', 0)} cas \texttt{{normal}} et "
            rf"{label_counts.get('suspicion_opacite', 0)} cas \texttt{{suspicion\_opacite}}.",
            "",
            "Les memes images sont utilisees pour comparer les configurations, ce qui rend la comparaison defendable malgre la taille limitee de l'echantillon.",
            "",
            r"\section{Configurations comparees}",
            r"\begin{itemize}",
            *[rf"\item \texttt{{{_latex_escape(model_name)}}} : {_latex_escape(_model_label(model_name))}." for model_name in model_names],
            r"\end{itemize}",
            "",
            r"\section{Resultats quantitatifs}",
            r"\begin{table}[H]",
            r"\centering",
            r"\small",
            r"\begin{tabular}{lrrrrrrrrr}",
            r"\toprule",
            r"Modele & n & Acc. & F1 & Sens. & Spec. & Inc. \% & FP & FN & Lat. ms \\",
            r"\midrule",
            *metric_rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Synthese des metriques principales.}",
            r"\end{table}",
            "",
            "\n".join(figures_tex),
            "",
            r"\section{Analyse qualitative}",
            rf"La baseline Gemma obtient une sensibilite elevee ({baseline.get('sensitivity', 'n/a')}) mais une specificite faible "
            rf"({baseline.get('specificity', 'n/a')}). Elle detecte beaucoup de cas positifs, mais produit aussi "
            rf"{baseline.get('false_positives', 'n/a')} faux positifs sur l'echantillon.",
            "",
            rf"La version MedGemma renforcee ameliore le macro-F1 ({reinforced.get('macro_f1', 'n/a')}) et reduit les faux positifs "
            rf"({reinforced.get('false_positives', 'n/a')}). Le gain de macro-F1 observe est de {macro_gain}, avec une reduction de {fp_delta} faux positifs. "
            "Le compromis principal est une latence plus elevee et un taux d'incertitude plus important.",
            "",
            r"Cette incertitude est acceptable dans le cadre du sujet : elle montre que le systeme peut refuser de conclure lorsque les prompts ne convergent pas.",
            "",
            r"\section{Points forts}",
            r"\begin{itemize}",
            r"\item Chaine complete : preparation RSNA, pretraitement, inference, JSON, SQLite, dashboard et rapports.",
            r"\item Comparaison experimentale entre baseline et amelioration.",
            r"\item Resultats reproductibles a partir de \texttt{runs.sqlite}.",
            r"\item Warning medical et limites non cliniques clairement documentes.",
            r"\item Validite JSON mesuree et erreurs visibles.",
            r"\end{itemize}",
            "",
            r"\section{Limites assumees}",
            r"\begin{itemize}",
            r"\item L'evaluation porte sur 30 cas : c'est suffisant pour une preuve pedagogique, pas pour une validation clinique.",
            r"\item Les labels RSNA ne remplacent pas une expertise radiologique complete.",
            r"\item Les hallucinations ne sont pas annotees manuellement ; le compteur indique uniquement les hallucinations taggees.",
            r"\item Le fine-tuning LoRA a ete essaye mais non retenu, car il degradait la robustesse du format JSON.",
            r"\item Le systeme ne doit jamais etre presente comme un dispositif medical.",
            r"\end{itemize}",
            "",
            r"\section{Conclusion}",
            "Le projet atteint l'objectif principal : fournir une preuve d'ingenierie responsable pour une IA medicale multimodale. "
            "La progression de la baseline vers MedGemma renforcee est mesurable, les erreurs sont documentees, les limites sont assumees, "
            "et la demonstration peut etre faite sans relancer les modeles grace aux runs sauvegardes.",
            "",
            r"\end{document}",
            "",
        ]
    )
    path.write_text(tex, encoding="utf-8")


def main() -> int:
    init_db(DB_PATH)
    rows = [dict(row) for row in fetch_runs(db_path=DB_PATH)]
    if not rows:
        print("[WARN] Aucun run dans la base. Lance scripts/run_eval.py d'abord.")
        return 1

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = REPORT_DIR / "report.csv"
    md_path = REPORT_DIR / "report.md"
    qualitative_path = REPORT_DIR / "rapport_qualitatif.md"
    latex_path = REPORT_DIR / "rapport_final.tex"

    by_case: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("case_id"):
            by_case[str(row["case_id"])].append(row)

    model_names = []
    for row in rows:
        model_name = row.get("model_name")
        if model_name and model_name not in model_names:
            model_names.append(model_name)

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        header = ["case_id", "image_path", "ground_truth", "patient_summary"]
        for model_name in model_names:
            header += [
                f"pred_{model_name}",
                f"conf_{model_name}",
                f"json_valid_{model_name}",
                f"reason_{model_name}",
                f"error_{model_name}",
            ]
        writer.writerow(header)

        for case_id, case_runs in sorted(by_case.items()):
            gt = next((run.get("ground_truth") for run in case_runs if run.get("ground_truth")), "unknown")
            image_path = case_runs[0].get("image_path", "")
            patient_summary = case_runs[0].get("true_label") or gt
            row = [case_id, image_path, gt, patient_summary]
            by_model = {run.get("model_name"): run for run in case_runs}
            for model_name in model_names:
                run = by_model.get(model_name)
                if run:
                    error_kind = "erreur_technique" if run.get("error_message") else (
                        auto_error_type(run["pred_class"], gt, bool(run["json_valid"])) or ""
                    )
                    row += [
                        run.get("pred_class", ""),
                        run.get("confidence", ""),
                        bool(run.get("json_valid")),
                        run.get("reason", ""),
                        error_kind,
                    ]
                else:
                    row += ["", "", "", "", ""]
            writer.writerow(row)
    print(f"[OK] CSV: {csv_path}")

    by_model_runs: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("model_name") and row.get("ground_truth"):
            by_model_runs[str(row["model_name"])].append(row)

    metrics_by_model = _metric_table(by_model_runs)
    label_counts = _unique_label_counts(rows)
    error_comments = [dict(item) for item in fetch_errors(db_path=DB_PATH)]
    selected_comments = select_commented_cases(rows)

    infographic_paths = {
        "metrics": REPORT_DIR / "infographie_performances.png",
        "errors": REPORT_DIR / "infographie_erreurs.png",
        "latency": REPORT_DIR / "infographie_latence_json.png",
    }
    _save_metric_comparison(metrics_by_model, infographic_paths["metrics"])
    _save_error_breakdown(metrics_by_model, infographic_paths["errors"])
    _save_latency_json(metrics_by_model, infographic_paths["latency"])
    confusion_paths = _save_confusion_matrices(by_model_runs, REPORT_DIR)
    print(f"[OK] Infographies: {REPORT_DIR}")

    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("# Rapport - Assistant Radiologue Virtuel\n\n")
        handle.write(f"> **{WARNING_TEXT}**\n\n")
        handle.write("## Contexte du projet\n\n")
        handle.write(
            "Prototype pedagogique d'assistant radiologique virtuel pour radiographies thoraciques frontales. "
            "Le systeme produit un JSON structure avec trois classes possibles : `normal`, "
            "`suspicion_opacite`, `incertain`.\n\n"
        )
        handle.write("## Dataset utilise\n\n")
        handle.write(
            f"- Source : RSNA Pneumonia Detection Challenge (Kaggle)\n"
            f"- Labels source : `{LABELS_CSV}`\n"
            f"- Cas evalues : {len(by_case)}\n"
            f"- Repartition : normal={label_counts.get('normal', 0)}, "
            f"suspicion_opacite={label_counts.get('suspicion_opacite', 0)}\n\n"
        )
        handle.write("## Configurations comparees\n\n")
        for model_name in model_names:
            handle.write(f"- `{model_name}`\n")
        handle.write("\n## Infographies\n\n")
        handle.write(f"![Comparaison des performances]({infographic_paths['metrics'].name})\n\n")
        handle.write(f"![Typologie des erreurs]({infographic_paths['errors'].name})\n\n")
        handle.write(f"![Latence et validite JSON]({infographic_paths['latency'].name})\n\n")
        for path in confusion_paths:
            handle.write(f"![{path.stem}]({path.name})\n\n")
        handle.write("\n## Tableau des metriques\n\n")
        handle.write(
            "| Configuration | n | Accuracy | Macro-F1 | Sensibilite | Specificite | "
            "Incertain % | JSON valide % | FP | FN | JSON invalides | Hallucinations taggees | Latence moyenne (ms) |\n"
        )
        handle.write("|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
        confusion_blocks = []
        for model_name, runs_m in by_model_runs.items():
            metrics = metrics_by_model[model_name]
            handle.write(
                f"| {model_name} | {metrics['n']} | {metrics['accuracy']} | {metrics['macro_f1']} | "
                f"{metrics['sensitivity']} | {metrics['specificity']} | {metrics['uncertain_pct']} | "
                f"{metrics['json_valid_pct']} | {metrics['false_positives']} | {metrics['false_negatives']} | "
                f"{metrics['json_invalid_cases']} | {metrics['hallucination_flags']} | {metrics['latency_mean_ms']} |\n"
            )
            matrix, classes = confusion_matrix(runs_m)
            confusion_blocks.append((model_name, matrix.tolist(), classes))

        handle.write("\n## Matrices de confusion\n\n")
        for model_name, matrix, classes in confusion_blocks:
            handle.write(f"### {model_name}\n\n")
            handle.write(f"- Classes : {classes}\n")
            handle.write(f"- Matrice : `{matrix}`\n\n")

        handle.write("## Synthese d'erreurs\n\n")
        global_errors = summarize_errors(rows)
        for key, value in global_errors.items():
            handle.write(f"- {key}: {value}\n")
        handle.write("\n")

        handle.write(f"## Cas commentes ({len(selected_comments)})\n\n")
        if not selected_comments:
            handle.write("Aucun cas commente disponible.\n\n")
        else:
            grouped_comments: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
            for run in selected_comments:
                grouped_comments[_run_bucket(run)].append(run)
            labels = {
                "false_negative": "Faux negatifs",
                "false_positive": "Faux positifs",
                "uncertain": "Incertains",
                "json_invalid": "JSON invalides",
                "technical_errors": "Erreurs techniques",
                "success": "Reussites",
                "other": "Autres",
            }
            for bucket in (
                "false_negative",
                "false_positive",
                "uncertain",
                "json_invalid",
                "technical_errors",
                "success",
                "other",
            ):
                entries = grouped_comments.get(bucket, [])
                if not entries:
                    continue
                handle.write(f"### {labels[bucket]}\n\n")
                for run in entries:
                    handle.write(_comment_line(run) + "\n")
                handle.write("\n")

        handle.write("## Registre d'erreurs\n\n")
        if not error_comments:
            handle.write("Aucun commentaire manuel enregistre.\n\n")
        else:
            for item in error_comments[:20]:
                handle.write(
                    f"- run_id={item.get('run_id')} case_id=`{item.get('case_id')}` "
                    f"type=`{item.get('error_type')}` commentaire=`{item.get('comment')}`\n"
                )
            handle.write("\n")

        handle.write("## Limites\n\n")
        handle.write(
            "- Prototype pedagogique, non destine au diagnostic.\n"
            "- La classe `incertain` est une sortie de securite du systeme, pas un label Kaggle.\n"
            "- Les performances dependent fortement du chargement correct des modeles Hugging Face et de CUDA.\n"
            "- Le dashboard peut etre montre a partir de runs pre-calcules sans inference live.\n\n"
        )
        handle.write("## Conformite et securite\n\n")
        handle.write(
            "- Voir `ETHICS.md` pour le cadrage non clinique, les garde-fous et les limites.\n"
            "- Voir `DATASET_USAGE.md` pour la provenance des donnees et les conditions d'usage.\n\n"
        )
        handle.write("## Conclusion\n\n")
        handle.write(
            "Le projet demontre une chaine d'ingenierie complete : preparation du dataset RSNA, "
            "pretraitement DICOM/PNG, sortie JSON structuree, journalisation SQLite, evaluation multi-configurations "
            "et analyse d'erreurs, sans revendiquer un usage clinique.\n"
        )
    print(f"[OK] Markdown: {md_path}")

    _write_qualitative_report(
        path=qualitative_path,
        rows=rows,
        by_case=by_case,
        model_names=model_names,
        metrics_by_model=metrics_by_model,
        label_counts=label_counts,
        infographic_paths=infographic_paths,
        confusion_paths=confusion_paths,
    )
    print(f"[OK] Rapport qualitatif: {qualitative_path}")

    _write_latex_report(
        path=latex_path,
        model_names=model_names,
        metrics_by_model=metrics_by_model,
        label_counts=label_counts,
        by_case=by_case,
        confusion_paths=confusion_paths,
        infographic_paths=infographic_paths,
    )
    print(f"[OK] LaTeX: {latex_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
