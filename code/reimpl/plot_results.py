"""Aggregate LoRA NLU experiment results and make simple ablation plots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize LoRA NLU experiment results.")
    parser.add_argument("--results_root", default="results/nlu", help="Directory containing experiment subfolders.")
    parser.add_argument("--output_dir", default=None, help="Defaults to results_root.")
    parser.add_argument("--paper_tasks", default="sst2,mrpc,cola,rte", help="Comma-separated tasks for paper-style table.")
    parser.add_argument("--table_name", default="paper_style_4task_table", help="Base filename for paper-style table.")
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
        target_modules = config.get("target_modules", [])
        row = {
            "experiment": exp_dir.name,
            "method": metrics.get("method", config.get("method", "lora")),
            "task_name": metrics.get("task_name", config.get("task_name")),
            "model_name": metrics.get("model_name", config.get("model_name")),
            "primary_metric": metrics.get("primary_metric"),
            "primary_metric_value": metrics.get("primary_metric_value"),
            "accuracy": metrics.get("accuracy"),
            "f1": metrics.get("f1"),
            "matthews_correlation": metrics.get("matthews_correlation"),
            "eval_loss": metrics.get("eval_loss"),
            "lora_r": config.get("lora_r"),
            "lora_alpha": config.get("lora_alpha"),
            "lora_dropout": config.get("lora_dropout"),
            "adapter_size": config.get("adapter_size"),
            "adapter_location": config.get("adapter_location"),
            "target_modules": ",".join(target_modules) if isinstance(target_modules, list) else target_modules,
            "trainable_parameters": params.get("trainable_parameters"),
            "total_parameters": params.get("total_parameters"),
            "trainable_ratio": params.get("trainable_ratio"),
        }
        rows.append(row)
    return rows


def method_label(row: pd.Series) -> str:
    method = row.get("method", "lora")
    if method == "ft":
        return "RoBbase (FT)"
    if method == "bitfit":
        return "RoBbase (BitFit)"
    if method == "adapter":
        size = row.get("adapter_size")
        if pd.notna(size):
            try:
                size = int(size)
            except (TypeError, ValueError):
                pass
            return f"RoBbase (Adapter, size={size})"
        return "RoBbase (Adapter)"
    if method == "lora":
        rank = row.get("lora_r")
        try:
            rank = int(rank)
        except (TypeError, ValueError):
            rank = None
        return f"RoBbase (LoRA r={rank})" if rank is not None else "RoBbase (LoRA)"
    return str(method)


def make_paper_style_table(df: pd.DataFrame, output_dir: Path, tasks: list[str], table_name: str) -> None:
    if df.empty:
        return
    work = df.copy()
    work["task_name"] = work["task_name"].astype(str).str.lower()
    work = work[work["task_name"].isin(tasks)]
    work = work.dropna(subset=["primary_metric_value"])
    if work.empty:
        return
    work["method_label"] = work.apply(method_label, axis=1)
    work = work.sort_values(["method_label", "task_name", "experiment"])
    work = work.groupby(["method_label", "task_name"], as_index=False).tail(1)

    pivot = work.pivot(index="method_label", columns="task_name", values="primary_metric_value")
    for task in tasks:
        if task not in pivot.columns:
            pivot[task] = pd.NA
    pivot = pivot[tasks]
    pivot["Avg4"] = pivot[tasks].mean(axis=1, skipna=True)

    params = (
        work.groupby("method_label")["trainable_parameters"]
        .median()
        .apply(lambda value: f"{value / 1_000_000:.2f}M" if pd.notna(value) else "")
    )
    table = pivot.copy()
    table.insert(0, "# Trainable Parameters", params)
    table = table.reset_index().rename(columns={"method_label": "Model & Method"})
    table.to_csv(output_dir / f"{table_name}.csv", index=False)

    markdown_path = output_dir / f"{table_name}.md"
    with markdown_path.open("w", encoding="utf-8") as f:
        f.write("# Four-Task Paper-Style Result Table\n\n")
        f.write("`Avg4` is the average over the selected four tasks, not the original paper's 8-task GLUE average.\n\n")
        headers = list(table.columns)
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for _, row in table.iterrows():
            values = []
            for header in headers:
                value = row[header]
                if isinstance(value, float):
                    values.append(f"{value:.4f}")
                elif pd.isna(value):
                    values.append("")
                else:
                    values.append(str(value))
            f.write("| " + " | ".join(values) + " |\n")


def make_plots(df: pd.DataFrame, output_dir: Path) -> None:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    clean = df.dropna(subset=["primary_metric_value"])
    if clean.empty:
        return

    rank_df = clean.dropna(subset=["lora_r"]).sort_values("lora_r")
    if not rank_df.empty:
        plt.figure(figsize=(6, 4))
        for target, group in rank_df.groupby("target_modules", dropna=False):
            plt.plot(group["lora_r"], group["primary_metric_value"], marker="o", label=str(target))
        plt.xlabel("LoRA rank r")
        plt.ylabel("Primary metric")
        plt.title("LoRA rank ablation")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(figures_dir / "metric_vs_rank.png", dpi=200)
        plt.close()

    param_df = clean.dropna(subset=["trainable_parameters"])
    if not param_df.empty:
        plt.figure(figsize=(6, 4))
        plt.scatter(param_df["trainable_parameters"], param_df["primary_metric_value"])
        for _, row in param_df.iterrows():
            plt.annotate(row["experiment"], (row["trainable_parameters"], row["primary_metric_value"]), fontsize=8)
        plt.xlabel("Trainable parameters")
        plt.ylabel("Primary metric")
        plt.title("Parameter efficiency")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(figures_dir / "trainable_params_vs_metric.png", dpi=200)
        plt.close()


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root)
    output_dir = Path(args.output_dir) if args.output_dir else results_root
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_rows(results_root)
    df = pd.DataFrame(rows)
    summary_path = output_dir / "summary_table.csv"
    df.to_csv(summary_path, index=False)
    if not df.empty:
        make_plots(df, output_dir)
        tasks = [task.strip().lower() for task in args.paper_tasks.split(",") if task.strip()]
        make_paper_style_table(df, output_dir, tasks, args.table_name)
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
