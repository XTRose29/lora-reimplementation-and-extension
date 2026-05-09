"""Evaluate a checkpoint produced by train_my_lora_nlu.py."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

try:
    from .my_lora import load_trainable_state
    from .my_modeling import (
        compute_task_metrics,
        get_task_metadata,
        inject_adapter_into_encoder,
        inject_lora_into_encoder,
        load_lora_config,
        load_tokenizer_and_sequence_classifier,
    )
    from .train_my_lora_nlu import preprocess_dataset
except ImportError:
    from my_lora import load_trainable_state
    from my_modeling import (
        compute_task_metrics,
        get_task_metadata,
        inject_adapter_into_encoder,
        inject_lora_into_encoder,
        load_lora_config,
        load_tokenizer_and_sequence_classifier,
    )
    from train_my_lora_nlu import preprocess_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate our custom LoRA NLU checkpoint.")
    parser.add_argument("--checkpoint_dir", required=True, help="Directory with trainable_state.pt and lora_config.json.")
    parser.add_argument("--output_dir", default=None, help="Defaults to checkpoint parent.")
    parser.add_argument("--task_name", default=None, help="Override task name from lora_config.json.")
    parser.add_argument("--model_name", default=None, help="Override model name from lora_config.json.")
    parser.add_argument("--split", default=None, help="Dataset split. Defaults to task validation split.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


@torch.no_grad()
def run_eval(model, dataloader, task_name: str, device: torch.device):
    model.eval()
    losses = []
    predictions = []
    labels = []
    logits_dump = []
    is_regression = task_name.lower() == "stsb"

    for batch in dataloader:
        batch = {key: value.to(device) for key, value in batch.items()}
        outputs = model(**batch)
        losses.append(float(outputs.loss.detach().cpu()))
        logits = outputs.logits.detach().cpu()
        label_batch = batch["labels"].detach().cpu()
        logits_dump.extend(logits.numpy().tolist())
        if is_regression:
            pred_batch = logits.squeeze(-1).numpy().tolist()
        else:
            pred_batch = torch.argmax(logits, dim=-1).numpy().tolist()
        predictions.extend(pred_batch)
        labels.extend(label_batch.numpy().tolist())

    metrics = compute_task_metrics(task_name, predictions, labels)
    metrics["eval_loss"] = float(np.mean(losses)) if losses else 0.0
    return metrics, predictions, labels, logits_dump


def main() -> None:
    args = parse_args()
    checkpoint_dir = Path(args.checkpoint_dir)
    config = load_lora_config(checkpoint_dir)
    task_name = (args.task_name or config["task_name"]).lower()
    model_name = args.model_name or config["model_name"]
    output_dir = Path(args.output_dir) if args.output_dir else checkpoint_dir.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    metadata = get_task_metadata(task_name)
    split = args.split or str(metadata["validation_split"])
    tokenizer, model = load_tokenizer_and_sequence_classifier(model_name, task_name)
    method = config.get("method", "lora")
    if method == "lora":
        inject_lora_into_encoder(
            model,
            r=int(config["lora_r"]),
            alpha=float(config["lora_alpha"]),
            dropout=float(config["lora_dropout"]),
            target_modules=tuple(config["target_modules"]),
        )
    elif method == "adapter":
        inject_adapter_into_encoder(
            model,
            adapter_size=int(config["adapter_size"]),
            dropout=float(config.get("adapter_dropout", 0.0)),
            location=str(config.get("adapter_location", "output")),
        )
    elif method in {"ft", "bitfit"}:
        pass
    else:
        raise ValueError(f"Unsupported method in checkpoint config: {method}")
    load_trainable_state(model, checkpoint_dir / "trainable_state.pt", map_location="cpu")
    model.to(device)

    raw_dataset = load_dataset("glue", task_name)[split]
    if args.max_eval_samples is not None:
        raw_dataset = raw_dataset.select(range(min(args.max_eval_samples, len(raw_dataset))))
    encoded = preprocess_dataset(raw_dataset, tokenizer, task_name, args.max_length)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    dataloader = DataLoader(encoded, batch_size=args.batch_size, shuffle=False, collate_fn=data_collator)

    metrics, predictions, labels, logits_dump = run_eval(model, dataloader, task_name, device)
    metrics.update({"task_name": task_name, "model_name": model_name, "method": method, "split": split})
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    with (output_dir / "eval.txt").open("w", encoding="utf-8") as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
    with (output_dir / f"{split}_predictions.jsonl").open("w", encoding="utf-8") as f:
        for idx, (prediction, label, logits) in enumerate(zip(predictions, labels, logits_dump)):
            f.write(json.dumps({"idx": idx, "prediction": prediction, "label": label, "logits": logits}) + "\n")

    print(f"Evaluation complete. {metrics['primary_metric']}: {metrics['primary_metric_value']:.4f}")
    print(f"Outputs: {output_dir}")


if __name__ == "__main__":
    main()
