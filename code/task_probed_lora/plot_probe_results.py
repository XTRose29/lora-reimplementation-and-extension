"""Aggregate Task-Probed LoRA results and make small plots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Task-Probed LoRA NLU results.")
    parser.add_argument("--results_root", default="newimpl/results", help="Directory containing experiment folders.")
    parser.add_argument("--output_dir", default=None, help="Defaults to results_root.")
    return parser.parse_args()


def load_json(path: Path):
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def collect_rows(results_root: Path):
    rows = []
    for metrics_path in sorted(results_root.rglob("metrics.json")):
        exp_dir = metrics_path.parent
        metrics = load_json(metrics_path)
        config = load_json(exp_dir / "train_config.json")
        params = load_json(exp_dir / "parameter_count.json")
        if metrics.get("method") != "task_probed_lora" and config.get("method") != "task_probed_lora":
            continue
        rows.append(
            {
                "experiment": str(exp_dir.relative_to(results_root)),
                "task_name": metrics.get("task_name", config.get("task_name")),
                "model_name": metrics.get("model_name", config.get("model_name")),
                "method": metrics.get("method", config.get("method")),
                "probe_strategy": metrics.get("probe_strategy", config.get("probe_strategy")),
                "top_k": metrics.get("top_k", config.get("top_k")),
                "candidate_module_count": metrics.get("candidate_module_count", config.get("candidate_module_count")),
                "selected_module_count": metrics.get("selected_module_count", config.get("selected_module_count")),
                "primary_metric": metrics.get("primary_metric"),
                "primary_metric_value": metrics.get("primary_metric_value"),
                "accuracy": metrics.get("accuracy"),
                "f1": metrics.get("f1"),
                "matthews_correlation": metrics.get("matthews_correlation"),
                "eval_loss": metrics.get("eval_loss"),
                "trainable_parameters": params.get("trainable_parameters"),
                "trainable_ratio": params.get("trainable_ratio"),
            }
        )
    return rows


def make_metric_plot(df: pd.DataFrame, output_dir: Path) -> None:
    clean = df.dropna(subset=["primary_metric_value"])
    if clean.empty:
        return
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    for task, group in clean.groupby("task_name"):
        group = group.sort_values(["selected_module_count", "probe_strategy"])
        plt.figure(figsize=(7, 4))
        plt.bar(group["probe_strategy"].astype(str), group["primary_metric_value"])
        plt.xlabel("Probe strategy")
        plt.ylabel(str(group["primary_metric"].iloc[0]))
        plt.title(f"Task-Probed LoRA on {task}")
        plt.xticks(rotation=25, ha="right")
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(figures_dir / f"{task}_metric_by_strategy.png", dpi=200)
        plt.close()


def make_probe_heatmaps(results_root: Path, output_dir: Path) -> None:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    for score_path in sorted(results_root.rglob("probe_scores.json")):
        rows = load_json(score_path)
        if not rows:
            continue
        df = pd.DataFrame(rows)
        if "layer" not in df.columns or "kind" not in df.columns:
            continue
        df = df.dropna(subset=["layer"])
        if df.empty:
            continue
        for score_name in ["gradient_norm", "activation_norm", "actgrad_score"]:
            if score_name not in df.columns:
                continue
            pivot = df.pivot_table(index="kind", columns="layer", values=score_name, aggfunc="mean")
            if pivot.empty:
                continue
            plt.figure(figsize=(8, max(2.5, 0.7 * len(pivot))))
            plt.imshow(pivot.values, aspect="auto", cmap="viridis")
            plt.colorbar(label=score_name)
            plt.yticks(range(len(pivot.index)), pivot.index)
            plt.xticks(range(len(pivot.columns)), [int(col) for col in pivot.columns])
            plt.xlabel("Layer")
            plt.title(f"{score_path.parent.name}: {score_name}")
            plt.tight_layout()
            rel_name = "_".join(score_path.parent.relative_to(results_root).parts)
            plt.savefig(figures_dir / f"{rel_name}_{score_name}_heatmap.png", dpi=200)
            plt.close()


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root)
    output_dir = Path(args.output_dir) if args.output_dir else results_root
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_rows(results_root)
    df = pd.DataFrame(rows)
    summary_path = output_dir / "probe_summary_table.csv"
    df.to_csv(summary_path, index=False)
    if not df.empty:
        make_metric_plot(df, output_dir)
        make_probe_heatmaps(results_root, output_dir)
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
