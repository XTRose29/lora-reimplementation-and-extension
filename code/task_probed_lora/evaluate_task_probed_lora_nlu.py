"""Evaluate a Task-Probed LoRA checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

from newimpl.probe_lora import inject_lora_into_selected_modules
from reimpl.evaluate_my_lora_nlu import run_eval
from reimpl.my_lora import load_trainable_state
from reimpl.my_modeling import get_task_metadata, load_lora_config, load_tokenizer_and_sequence_classifier
from reimpl.train_my_lora_nlu import preprocess_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a Task-Probed LoRA checkpoint.")
    parser.add_argument("--checkpoint_dir", required=True, help="Directory with trainable_state.pt and lora_config.json.")
    parser.add_argument("--output_dir", default=None, help="Defaults to checkpoint parent.")
    parser.add_argument("--task_name", default=None)
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_dir = Path(args.checkpoint_dir)
    config = load_lora_config(checkpoint_dir)
    if config.get("method") != "task_probed_lora":
        raise ValueError(f"Expected method=task_probed_lora, got {config.get('method')!r}")

    task_name = (args.task_name or config["task_name"]).lower()
    model_name = args.model_name or config["model_name"]
    output_dir = Path(args.output_dir) if args.output_dir else checkpoint_dir.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    metadata = get_task_metadata(task_name)
    split = args.split or str(metadata["validation_split"])
    tokenizer, model = load_tokenizer_and_sequence_classifier(model_name, task_name)
    selected_modules = list(config["selected_modules"])
    inject_lora_into_selected_modules(
        model,
        selected_module_names=selected_modules,
        r=int(config["lora_r"]),
        alpha=float(config["lora_alpha"]),
        dropout=float(config["lora_dropout"]),
    )
    load_trainable_state(model, checkpoint_dir / "trainable_state.pt", map_location="cpu")
    model.to(device)

    raw_dataset = load_dataset("glue", task_name)[split]
    if args.max_eval_samples is not None:
        raw_dataset = raw_dataset.select(range(min(args.max_eval_samples, len(raw_dataset))))
    encoded = preprocess_dataset(raw_dataset, tokenizer, task_name, args.max_length)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    dataloader = DataLoader(encoded, batch_size=args.batch_size, shuffle=False, collate_fn=data_collator)

    metrics, predictions, labels, logits_dump = run_eval(model, dataloader, task_name, device)
    metrics.update(
        {
            "task_name": task_name,
            "model_name": model_name,
            "method": "task_probed_lora",
            "probe_strategy": config.get("probe_strategy"),
            "top_k": config.get("top_k"),
            "selected_module_count": len(selected_modules),
            "split": split,
        }
    )
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    with (output_dir / "eval.txt").open("w", encoding="utf-8") as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
    with (output_dir / f"{split}_predictions.jsonl").open("w", encoding="utf-8") as f:
        for idx, (prediction, label, logits) in enumerate(zip(predictions, labels, logits_dump)):
            f.write(json.dumps({"idx": idx, "prediction": prediction, "label": label, "logits": logits}) + "\n")

    print(
        "Evaluation complete. "
        f"{metrics['primary_metric']}: {metrics['primary_metric_value']:.4f}, "
        f"strategy={metrics['probe_strategy']}"
    )
    print(f"Outputs: {output_dir}")


if __name__ == "__main__":
    main()
