"""Aggregate GLUE reliability runs into CSV and Markdown tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FIELDS = [
    "run",
    "task_name",
    "model_name",
    "method",
    "lora_placement",
    "lora_r",
    "lora_alpha",
    "lora_dropout",
    "max_train_samples",
    "max_train_questions",
    "max_eval_questions",
    "num_negative_candidates",
    "layer_indices",
    "trainable_parameters",
    "train_candidate_examples",
    "eval_candidate_examples",
    "primary_metric",
    "primary_metric_value",
    "answer_selection_accuracy",
    "mrr",
    "matthews_correlation",
    "f1",
    "accuracy",
    "ece",
    "calibration_accuracy",
    "nll",
    "brier",
    "mean_confidence",
    "abstention_rate",
    "coverage",
    "selective_accuracy",
    "ood_mean_confidence",
    "ood_abstention_rate",
]


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def collect_run(run_dir: Path):
    metrics_path = run_dir / "metrics.json"
    config_path = run_dir / "train_config.json"
    params_path = run_dir / "parameter_count.json"
    if not metrics_path.exists() or not config_path.exists():
        return None
    metrics = load_json(metrics_path)
    config = load_json(config_path)
    params = load_json(params_path) if params_path.exists() else {}
    id_metrics = metrics.get("final_id", metrics)
    ood_metrics = metrics.get("ood", {})
    return {
        "run": str(run_dir.name),
        "task_name": config.get("task_name"),
        "model_name": config.get("model_name"),
        "method": config.get("method"),
        "lora_placement": config.get("lora_placement"),
        "lora_r": config.get("lora_r"),
        "lora_alpha": config.get("lora_alpha"),
        "lora_dropout": config.get("lora_dropout"),
        "max_train_samples": config.get("max_train_samples"),
        "max_train_questions": config.get("max_train_questions"),
        "max_eval_questions": config.get("max_eval_questions"),
        "num_negative_candidates": config.get("num_negative_candidates"),
        "layer_indices": config.get("layer_indices"),
        "trainable_parameters": params.get("trainable_parameters"),
        "train_candidate_examples": config.get("train_candidate_examples"),
        "eval_candidate_examples": config.get("eval_candidate_examples"),
        "primary_metric": id_metrics.get("primary_metric", config.get("primary_metric")),
        "primary_metric_value": id_metrics.get("primary_metric_value"),
        "answer_selection_accuracy": id_metrics.get("answer_selection_accuracy"),
        "mrr": id_metrics.get("mrr"),
        "matthews_correlation": id_metrics.get("matthews_correlation"),
        "f1": id_metrics.get("f1"),
        "accuracy": id_metrics.get("accuracy"),
        "ece": id_metrics.get("ece"),
        "calibration_accuracy": id_metrics.get("calibration_accuracy"),
        "nll": id_metrics.get("nll"),
        "brier": id_metrics.get("brier"),
        "mean_confidence": id_metrics.get("mean_confidence"),
        "abstention_rate": id_metrics.get("abstention_rate"),
        "coverage": id_metrics.get("coverage"),
        "selective_accuracy": id_metrics.get("selective_accuracy"),
        "ood_mean_confidence": ood_metrics.get("mean_confidence"),
        "ood_abstention_rate": ood_metrics.get("abstention_rate"),
    }


def fmt(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_root", default="results/reliability")
    parser.add_argument("--include_tasks", default="", help="Comma-separated task names to keep")
    parser.add_argument("--exclude_tasks", default="", help="Comma-separated task names to drop")
    args = parser.parse_args()
    root = Path(args.results_root)
    root.mkdir(parents=True, exist_ok=True)
    run_dirs = sorted(path for path in root.rglob("*") if path.is_dir() and (path / "metrics.json").exists())
    rows = [row for row in (collect_run(path) for path in run_dirs) if row]
    include_tasks = {task.strip().lower() for task in args.include_tasks.split(",") if task.strip()}
    exclude_tasks = {task.strip().lower() for task in args.exclude_tasks.split(",") if task.strip()}
    if include_tasks:
        rows = [row for row in rows if str(row.get("task_name", "")).lower() in include_tasks]
    if exclude_tasks:
        rows = [row for row in rows if str(row.get("task_name", "")).lower() not in exclude_tasks]

    csv_path = root / "summary_table.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    md_path = root / "summary_table.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("| " + " | ".join(FIELDS) + " |\n")
        f.write("| " + " | ".join(["---"] * len(FIELDS)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(fmt(row.get(field)) for field in FIELDS) + " |\n")

    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
