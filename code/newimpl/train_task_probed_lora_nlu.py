"""Train selective Task-Probed LoRA on GLUE NLU tasks."""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
from typing import Dict, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import DataCollatorWithPadding, get_linear_schedule_with_warmup

from newimpl.probe_lora import (
    add_normalized_scores,
    annotate_scores_with_selection,
    collect_probe_scores,
    find_candidate_linear_modules,
    inject_lora_into_selected_modules,
    make_unprobed_score_rows,
    select_modules_from_scores,
    write_json,
)
from reimpl.my_lora import count_trainable_parameters, mark_only_lora_and_head_as_trainable, save_trainable_state
from reimpl.my_modeling import (
    compute_task_metrics,
    get_task_metadata,
    load_tokenizer_and_sequence_classifier,
    save_lora_config,
)
from reimpl.train_my_lora_nlu import append_jsonl, preprocess_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Task-Probed LoRA on a GLUE NLU task.")
    parser.add_argument("--task_name", default="sst2", help="GLUE task name.")
    parser.add_argument("--model_name", default="roberta-base", help="Base Hugging Face model.")
    parser.add_argument("--output_dir", required=True, help="Experiment output directory.")
    parser.add_argument("--probe_strategy", default="actgrad", choices=["full", "random", "gradient", "activation", "actgrad"])
    parser.add_argument("--top_k", type=int, default=8, help="Number of candidate modules to receive LoRA.")
    parser.add_argument("--probe_samples", type=int, default=512)
    parser.add_argument("--probe_batch_size", type=int, default=16)
    parser.add_argument("--max_probe_batches", type=int, default=None)
    parser.add_argument("--candidate_scope", default="attention", choices=["attention", "attention_mlp"])
    parser.add_argument("--target_modules", default="query,value", help="Comma-separated attention module leaf names.")
    parser.add_argument("--lora_r", type=int, default=4)
    parser.add_argument("--lora_alpha", type=float, default=32.0)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
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
        else:
            pred_batch = torch.argmax(logits, dim=-1).numpy().tolist()
        predictions.extend(pred_batch)
        labels.extend(label_batch.numpy().tolist())

    metrics = compute_task_metrics(task_name, predictions, labels)
    metrics["eval_loss"] = float(np.mean(losses)) if losses else 0.0
    model.train()
    return metrics, predictions, labels


def save_checkpoint(model, tokenizer, checkpoint_dir: Path, method_config: Dict) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_trainable_state(model, checkpoint_dir / "trainable_state.pt")
    save_lora_config(checkpoint_dir, method_config)
    tokenizer.save_pretrained(checkpoint_dir)


def build_probe_selection(args, raw_dataset, task_name: str, device: torch.device, output_dir: Path):
    target_modules = tuple(part.strip() for part in args.target_modules.split(",") if part.strip())
    tokenizer, probe_model = load_tokenizer_and_sequence_classifier(args.model_name, task_name)
    candidate_modules = find_candidate_linear_modules(
        probe_model,
        target_modules=target_modules,
        candidate_scope=args.candidate_scope,
    )

    needs_probe = args.probe_strategy in {"gradient", "activation", "actgrad"}
    if needs_probe:
        probe_dataset = raw_dataset["train"]
        probe_count = min(args.probe_samples, len(probe_dataset))
        probe_dataset = probe_dataset.select(range(probe_count))
        encoded_probe = preprocess_dataset(probe_dataset, tokenizer, task_name, args.max_length)
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        probe_loader = DataLoader(encoded_probe, batch_size=args.probe_batch_size, shuffle=False, collate_fn=data_collator)
        score_rows = collect_probe_scores(
            probe_model,
            probe_loader,
            candidate_modules,
            device=device,
            max_probe_batches=args.max_probe_batches,
        )
    else:
        score_rows = add_normalized_scores(make_unprobed_score_rows(candidate_modules))

    selected_modules = select_modules_from_scores(
        score_rows,
        strategy=args.probe_strategy,
        top_k=args.top_k,
        seed=args.seed,
    )
    annotated_scores = annotate_scores_with_selection(score_rows, selected_modules)
    selection_payload = {
        "probe_strategy": args.probe_strategy,
        "top_k": args.top_k,
        "candidate_scope": args.candidate_scope,
        "target_modules": list(target_modules),
        "candidate_module_count": len(candidate_modules),
        "selected_module_count": len(selected_modules),
        "selected_modules": selected_modules,
    }
    write_json(output_dir / "probe_scores.json", annotated_scores)
    write_json(output_dir / "probe_selection.json", selection_payload)
    del probe_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return tokenizer, target_modules, candidate_modules, selected_modules, selection_payload


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    task_name = args.task_name.lower()
    metadata = get_task_metadata(task_name)
    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / "checkpoint"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    raw_dataset = load_dataset("glue", task_name)

    tokenizer, target_modules, candidate_modules, selected_modules, selection_payload = build_probe_selection(
        args,
        raw_dataset,
        task_name,
        device,
        output_dir,
    )

    _fresh_tokenizer, model = load_tokenizer_and_sequence_classifier(args.model_name, task_name)
    replaced_modules = inject_lora_into_selected_modules(
        model,
        selected_module_names=selected_modules,
        r=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
    )
    mark_only_lora_and_head_as_trainable(model)
    model.to(device)

    parameter_count = count_trainable_parameters(model)
    write_json(output_dir / "parameter_count.json", parameter_count)

    train_config = vars(args).copy()
    train_config.update(
        {
            "device": str(device),
            "method": "task_probed_lora",
            "target_modules": list(target_modules),
            "candidate_modules": candidate_modules,
            "selected_modules": selected_modules,
            "replaced_modules": replaced_modules,
            "validation_split": metadata["validation_split"],
            "primary_metric": metadata["primary_metric"],
            "parameter_count": parameter_count,
            **selection_payload,
        }
    )
    write_json(output_dir / "train_config.json", train_config)

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
        "method": "task_probed_lora",
        "model_name": args.model_name,
        "task_name": task_name,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "target_modules": list(target_modules),
        "candidate_scope": args.candidate_scope,
        "probe_strategy": args.probe_strategy,
        "top_k": args.top_k,
        "num_labels": int(metadata["num_labels"]),
        "candidate_modules": candidate_modules,
        "selected_modules": selected_modules,
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
        "method": "task_probed_lora",
        "probe_strategy": args.probe_strategy,
        "top_k": args.top_k,
        "candidate_module_count": len(candidate_modules),
        "selected_module_count": len(selected_modules),
        "best_epoch_metric": best_metric,
        **best_metrics,
    }
    write_json(output_dir / "metrics.json", final_metrics)
    with (output_dir / "eval.txt").open("w", encoding="utf-8") as f:
        for key, value in final_metrics.items():
            f.write(f"{key}: {value}\n")

    print(f"Training complete. Strategy={args.probe_strategy}, selected={len(selected_modules)}/{len(candidate_modules)}")
    print(f"Best {metadata['primary_metric']}: {best_metric:.4f}")
    print(f"Outputs: {output_dir}")


if __name__ == "__main__":
    main()
