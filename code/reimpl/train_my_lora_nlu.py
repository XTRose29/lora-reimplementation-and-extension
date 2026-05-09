"""Train our LoRA reimplementation on a GLUE NLU task."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import DataCollatorWithPadding, get_linear_schedule_with_warmup

try:
    from .my_adapter import mark_only_adapter_and_head_as_trainable
    from .my_lora import (
        count_trainable_parameters,
        mark_all_as_trainable,
        mark_only_bias_and_head_as_trainable,
        mark_only_lora_and_head_as_trainable,
        save_trainable_state,
    )
    from .my_modeling import (
        compute_task_metrics,
        get_task_metadata,
        inject_adapter_into_encoder,
        inject_lora_into_encoder,
        load_tokenizer_and_sequence_classifier,
        save_lora_config,
    )
except ImportError:
    from my_adapter import mark_only_adapter_and_head_as_trainable
    from my_lora import (
        count_trainable_parameters,
        mark_all_as_trainable,
        mark_only_bias_and_head_as_trainable,
        mark_only_lora_and_head_as_trainable,
        save_trainable_state,
    )
    from my_modeling import (
        compute_task_metrics,
        get_task_metadata,
        inject_adapter_into_encoder,
        inject_lora_into_encoder,
        load_tokenizer_and_sequence_classifier,
        save_lora_config,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a custom LoRA NLU model on GLUE.")
    parser.add_argument("--task_name", default="sst2", help="GLUE task name.")
    parser.add_argument("--model_name", default="distilroberta-base", help="Base Hugging Face model.")
    parser.add_argument("--output_dir", required=True, help="Experiment output directory.")
    parser.add_argument("--method", default="lora", choices=["lora", "ft", "bitfit", "adapter"])
    parser.add_argument("--lora_r", type=int, default=4)
    parser.add_argument("--lora_alpha", type=float, default=32.0)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--target_modules", default="query,value", help="Comma-separated attention module names.")
    parser.add_argument("--adapter_size", type=int, default=16)
    parser.add_argument("--adapter_dropout", type=float, default=0.0)
    parser.add_argument("--adapter_location", default="output", choices=["output", "attention_output", "both"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None, help="cuda, cpu, or leave unset for auto.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def append_jsonl(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


def preprocess_dataset(dataset, tokenizer, task_name: str, max_length: int):
    metadata = get_task_metadata(task_name)
    text_fields = tuple(metadata["text_fields"])

    def tokenize_batch(batch):
        texts = [batch[field] for field in text_fields]
        tokenized = tokenizer(*texts, truncation=True, max_length=max_length)
        tokenized["labels"] = batch["label"]
        return tokenized

    return dataset.map(tokenize_batch, batched=True, remove_columns=dataset.column_names)


@torch.no_grad()
def evaluate_model(model, dataloader, task_name: str, device: torch.device) -> Tuple[Dict, list, list]:
    model.eval()
    losses = []
    predictions = []
    labels = []
    is_regression = task_name.lower() == "stsb"

    for batch in dataloader:
        batch = {key: value.to(device) for key, value in batch.items()}
        outputs = model(**batch)
        losses.append(float(outputs.loss.detach().cpu()))
        logits = outputs.logits.detach().cpu()
        label_batch = batch["labels"].detach().cpu()
        if is_regression:
            pred_batch = logits.squeeze(-1).numpy().tolist()
            label_values = label_batch.numpy().tolist()
        else:
            pred_batch = torch.argmax(logits, dim=-1).numpy().tolist()
            label_values = label_batch.numpy().tolist()
        predictions.extend(pred_batch)
        labels.extend(label_values)

    metrics = compute_task_metrics(task_name, predictions, labels)
    metrics["eval_loss"] = float(np.mean(losses)) if losses else 0.0
    model.train()
    return metrics, predictions, labels


def save_checkpoint(model, tokenizer, checkpoint_dir: Path, method_config: Dict) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_trainable_state(model, checkpoint_dir / "trainable_state.pt")
    save_lora_config(checkpoint_dir, method_config)
    tokenizer.save_pretrained(checkpoint_dir)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    task_name = args.task_name.lower()
    metadata = get_task_metadata(task_name)
    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / "checkpoint"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    target_modules = tuple(part.strip() for part in args.target_modules.split(",") if part.strip())

    tokenizer, model = load_tokenizer_and_sequence_classifier(args.model_name, task_name)
    replaced_modules = []
    if args.method == "lora":
        replaced_modules = inject_lora_into_encoder(
            model,
            r=args.lora_r,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            target_modules=target_modules,
        )
        mark_only_lora_and_head_as_trainable(model)
    elif args.method == "adapter":
        replaced_modules = inject_adapter_into_encoder(
            model,
            adapter_size=args.adapter_size,
            dropout=args.adapter_dropout,
            location=args.adapter_location,
        )
        mark_only_adapter_and_head_as_trainable(model)
    elif args.method == "bitfit":
        mark_only_bias_and_head_as_trainable(model)
    elif args.method == "ft":
        mark_all_as_trainable(model)
    else:
        raise ValueError(f"Unsupported method: {args.method}")
    model.to(device)

    parameter_count = count_trainable_parameters(model)
    write_json(output_dir / "parameter_count.json", parameter_count)

    train_config = vars(args).copy()
    train_config.update(
        {
            "device": str(device),
            "method": args.method,
            "target_modules": list(target_modules),
            "replaced_modules": replaced_modules,
            "validation_split": metadata["validation_split"],
            "primary_metric": metadata["primary_metric"],
            "parameter_count": parameter_count,
        }
    )
    write_json(output_dir / "train_config.json", train_config)

    raw_dataset = load_dataset("glue", task_name)
    train_dataset = raw_dataset["train"]
    eval_dataset = raw_dataset[str(metadata["validation_split"])]
    if args.max_train_samples is not None:
        train_dataset = train_dataset.select(range(min(args.max_train_samples, len(train_dataset))))
    if args.max_eval_samples is not None:
        eval_dataset = eval_dataset.select(range(min(args.max_eval_samples, len(eval_dataset))))

    encoded_train = preprocess_dataset(train_dataset, tokenizer, task_name, args.max_length)
    encoded_eval = preprocess_dataset(eval_dataset, tokenizer, task_name, args.max_length)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_loader = DataLoader(encoded_train, batch_size=args.batch_size, shuffle=True, collate_fn=data_collator)
    eval_loader = DataLoader(encoded_eval, batch_size=args.batch_size, shuffle=False, collate_fn=data_collator)

    optimizer = AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    update_steps_per_epoch = max(1, int(np.ceil(len(train_loader) / args.gradient_accumulation_steps)))
    total_update_steps = max(1, args.epochs * update_steps_per_epoch)
    warmup_steps = int(args.warmup_ratio * total_update_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_update_steps,
    )

    method_config = {
        "method": args.method,
        "model_name": args.model_name,
        "task_name": task_name,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "target_modules": list(target_modules),
        "adapter_size": args.adapter_size,
        "adapter_dropout": args.adapter_dropout,
        "adapter_location": args.adapter_location,
        "num_labels": int(metadata["num_labels"]),
        "replaced_modules": replaced_modules,
    }

    best_metric = -float("inf")
    best_metrics = {}
    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    model.train()

    for epoch in range(1, args.epochs + 1):
        progress = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", leave=False)
        epoch_losses = []
        for step, batch in enumerate(progress, start=1):
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / args.gradient_accumulation_steps
            loss.backward()
            epoch_losses.append(float(outputs.loss.detach().cpu()))

            should_step = step % args.gradient_accumulation_steps == 0 or step == len(train_loader)
            if should_step:
                torch.nn.utils.clip_grad_norm_(
                    [param for param in model.parameters() if param.requires_grad],
                    max_norm=1.0,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                progress.set_postfix(loss=f"{np.mean(epoch_losses[-20:]):.4f}")

        eval_metrics, _, _ = evaluate_model(model, eval_loader, task_name, device)
        train_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        log_record = {"epoch": epoch, "global_step": global_step, "train_loss": train_loss, **eval_metrics}
        append_jsonl(output_dir / "train_log.jsonl", log_record)

        current_metric = float(eval_metrics["primary_metric_value"])
        if current_metric > best_metric:
            best_metric = current_metric
            best_metrics = dict(eval_metrics)
            save_checkpoint(model, tokenizer, checkpoint_dir, method_config)

    final_metrics = {
        "task_name": task_name,
        "model_name": args.model_name,
        "method": args.method,
        "best_epoch_metric": best_metric,
        **best_metrics,
    }
    write_json(output_dir / "metrics.json", final_metrics)
    with (output_dir / "eval.txt").open("w", encoding="utf-8") as f:
        for key, value in final_metrics.items():
            f.write(f"{key}: {value}\n")

    print(f"Training complete. Best {metadata['primary_metric']}: {best_metric:.4f}")
    print(f"Outputs: {output_dir}")


if __name__ == "__main__":
    main()
