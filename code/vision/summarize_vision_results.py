"""Summarize vision experiment results into a simple CSV table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize vision LoRA experiment results.")
    parser.add_argument("--results_root", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--table_name", default="summary_table")
    return parser.parse_args()


def load_json(path: Path):
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def collect_rows(results_root: Path):
    rows = []
    for metrics_path in sorted(results_root.glob("*/metrics.json")):
        exp_dir = metrics_path.parent
        metrics = load_json(metrics_path)
        config = load_json(exp_dir / "train_config.json")
        params = load_json(exp_dir / "parameter_count.json")
        trainable_parameters = params.get("trainable_parameters")
        rows.append(
            {
                "experiment": exp_dir.name,
                "task_name": metrics.get("task_name", config.get("task_name")),
                "method": metrics.get("method", config.get("method")),
                "model_name": metrics.get("model_name", config.get("model_name")),
                "accuracy": metrics.get("accuracy"),
                "eval_loss": metrics.get("eval_loss"),
                "trainable_parameters": trainable_parameters,
                "# Trainable Parameters": (
                    f"{float(trainable_parameters) / 1_000_000:.2f}M"
                    if trainable_parameters is not None
                    else ""
                ),
                "trainable_ratio": params.get("trainable_ratio"),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root)
    output_dir = Path(args.output_dir) if args.output_dir else results_root
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(collect_rows(results_root))
    df.to_csv(output_dir / f"{args.table_name}.csv", index=False)
    if not df.empty:
        preferred_columns = [
            "experiment",
            "task_name",
            "method",
            "accuracy",
            "# Trainable Parameters",
            "trainable_parameters",
            "trainable_ratio",
            "eval_loss",
            "model_name",
        ]
        df = df[[column for column in preferred_columns if column in df.columns]]
        markdown_path = output_dir / f"{args.table_name}.md"
        with markdown_path.open("w", encoding="utf-8") as f:
            f.write("# Vision LoRA Result Table\n\n")
            headers = list(df.columns)
            f.write("| " + " | ".join(headers) + " |\n")
            f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
            for _, row in df.iterrows():
                values = []
                for header in headers:
                    value = row.get(header, "")
                    if header == "accuracy" and pd.notna(value):
                        values.append(f"{float(value):.4f}")
                    elif header == "trainable_ratio" and pd.notna(value):
                        values.append(f"{float(value):.6f}")
                    elif header == "eval_loss" and pd.notna(value):
                        values.append(f"{float(value):.4f}")
                    elif pd.isna(value):
                        values.append("")
                    else:
                        values.append(str(value))
                f.write("| " + " | ".join(values) + " |\n")
    print(f"Wrote summary to {output_dir / f'{args.table_name}.csv'}")


if __name__ == "__main__":
    main()
