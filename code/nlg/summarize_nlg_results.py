from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FIELDS = [
    "run",
    "task_name",
    "dataset_task",
    "model_name",
    "method",
    "prompt_variant",
    "trainable_parameters",
    "max_train_examples",
    "max_eval_examples",
    "primary_metric",
    "primary_metric_value",
    "bleu",
    "rouge_l",
    "accuracy",
    "exact_match_rate",
    "fact_recall_mean",
    "parse_rate",
    "mean_confidence",
    "median_confidence",
    "confidence_correct_mean",
    "confidence_wrong_mean",
    "ece",
    "calibration_accuracy",
    "high_conf_wrong_rate",
    "low_conf_correct_rate",
    "abstention_rate",
    "coverage",
    "selective_accuracy",
    "selective_rouge_l",
    "mean_answer_tokens",
    "mean_response_tokens",
    "mean_sequence_logprob",
    "mean_answer_logprob",
    "abstained_by_text_rate",
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
    metrics = load_json(metrics_path).get("final_id", {})
    config = load_json(config_path)
    params = load_json(params_path) if params_path.exists() else {}
    return {
        "run": run_dir.name,
        "task_name": config.get("task_name"),
        "dataset_task": config.get("dataset_task"),
        "model_name": config.get("model_name"),
        "method": config.get("method"),
        "prompt_variant": config.get("prompt_variant"),
        "trainable_parameters": params.get("trainable_parameters"),
        "max_train_examples": config.get("max_train_examples"),
        "max_eval_examples": config.get("max_eval_examples"),
        "primary_metric": metrics.get("primary_metric"),
        "primary_metric_value": metrics.get("primary_metric_value"),
        "bleu": metrics.get("bleu"),
        "rouge_l": metrics.get("rouge_l"),
        "accuracy": metrics.get("accuracy"),
        "exact_match_rate": metrics.get("exact_match_rate"),
        "fact_recall_mean": metrics.get("fact_recall_mean"),
        "parse_rate": metrics.get("parse_rate"),
        "mean_confidence": metrics.get("mean_confidence"),
        "median_confidence": metrics.get("median_confidence"),
        "confidence_correct_mean": metrics.get("confidence_correct_mean"),
        "confidence_wrong_mean": metrics.get("confidence_wrong_mean"),
        "ece": metrics.get("ece"),
        "calibration_accuracy": metrics.get("calibration_accuracy"),
        "high_conf_wrong_rate": metrics.get("high_conf_wrong_rate"),
        "low_conf_correct_rate": metrics.get("low_conf_correct_rate"),
        "abstention_rate": metrics.get("abstention_rate"),
        "coverage": metrics.get("coverage"),
        "selective_accuracy": metrics.get("selective_accuracy"),
        "selective_rouge_l": metrics.get("selective_rouge_l"),
        "mean_answer_tokens": metrics.get("mean_answer_tokens"),
        "mean_response_tokens": metrics.get("mean_response_tokens"),
        "mean_sequence_logprob": metrics.get("mean_sequence_logprob"),
        "mean_answer_logprob": metrics.get("mean_answer_logprob"),
        "abstained_by_text_rate": metrics.get("abstained_by_text_rate"),
    }


def fmt(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_root", required=True)
    args = parser.parse_args()

    root = Path(args.results_root)
    run_dirs = sorted(path for path in root.rglob("*") if path.is_dir() and (path / "metrics.json").exists())
    rows = [row for row in (collect_run(path) for path in run_dirs) if row]

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
