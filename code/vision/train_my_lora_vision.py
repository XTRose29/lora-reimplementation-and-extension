"""Train our custom LoRA reimplementation on vision classification tasks."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
from datasets import load_dataset
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoImageProcessor, AutoModelForImageClassification, get_linear_schedule_with_warmup


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reimpl.my_lora import (  # noqa: E402
    MyLoRALinear,
    count_trainable_parameters,
    mark_all_as_trainable,
    mark_only_lora_and_head_as_trainable,
    save_trainable_state,
)


TASK_METADATA: Dict[str, Dict[str, object]] = {
    "cifar10": {
        "dataset_name": "cifar10",
        "train_split": "train",
        "eval_split": "test",
        "primary_metric": "accuracy",
    },
    "beans": {
        "dataset_name": "beans",
        "train_split": "train",
        "eval_split": "validation",
        "primary_metric": "accuracy",
    },
    "oxford_iiit_pet": {
        "dataset_name": "timm/oxford-iiit-pet",
        "train_split": "train",
        "eval_split": "test",
        "primary_metric": "accuracy",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a custom LoRA vision model.")
    parser.add_argument("--task_name", required=True, choices=sorted(TASK_METADATA))
    parser.add_argument("--model_name", default="google/vit-base-patch16-224-in21k")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--method", default="lora", choices=["lora", "ft"])
    parser.add_argument("--lora_r", type=int, default=4)
    parser.add_argument("--lora_alpha", type=float, default=32.0)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--target_modules", default="query,value")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
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


def get_task_metadata(task_name: str) -> Dict[str, object]:
    if task_name not in TASK_METADATA:
        raise ValueError(f"Unsupported task '{task_name}'")
    return TASK_METADATA[task_name]


def _get_parent_module(root: nn.Module, module_name: str) -> Tuple[nn.Module, str]:
    parts = module_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def _is_attention_target(module_name: str, module: nn.Module, target_modules: Sequence[str]) -> bool:
    if not isinstance(module, nn.Linear):
        return False
    leaf_name = module_name.rsplit(".", 1)[-1]
    if leaf_name not in set(target_modules):
        return False
    attention_markers = (".attention.", ".attn.", ".self_attn.")
    return any(marker in f".{module_name}." for marker in attention_markers)


def inject_lora_into_vision_encoder(
    model: nn.Module,
    r: int = 4,
    alpha: float = 32.0,
    dropout: float = 0.1,
    target_modules: Iterable[str] = ("query", "value"),
) -> List[str]:
    target_modules = tuple(target_modules)
    replaced: List[str] = []
    named_modules = list(model.named_modules())
    for module_name, module in named_modules:
        if not module_name:
            continue
        if not _is_attention_target(module_name, module, target_modules):
            continue
        parent, child_name = _get_parent_module(model, module_name)
        setattr(parent, child_name, MyLoRALinear.from_linear(module, r, alpha, dropout))
        replaced.append(module_name)

    if not replaced:
        examples = [
            name
            for name, module in named_modules
            if isinstance(module, nn.Linear) and name.rsplit(".", 1)[-1] in target_modules
        ][:10]
        raise ValueError(
            "No attention modules were replaced. "
            f"target_modules={target_modules}. Matching linear examples={examples}"
        )
    return replaced


def load_image_processor_and_model(model_name: str, task_name: str):
    metadata = get_task_metadata(task_name)
    raw_dataset = load_dataset(str(metadata["dataset_name"]))
    train_split = str(metadata["train_split"])
    label_column_name = detect_label_column_name(raw_dataset[train_split])
    label_feature = raw_dataset[train_split].features[label_column_name]
    label_names = list(label_feature.names)
    num_labels = len(label_names)
    image_processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
    model = AutoModelForImageClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label={i: name for i, name in enumerate(label_names)},
        label2id={name: i for i, name in enumerate(label_names)},
        ignore_mismatched_sizes=True,
    )
    return raw_dataset, image_processor, model, label_names, label_column_name


def detect_image_column_name(dataset) -> str:
    for column_name, feature in dataset.features.items():
        if feature.__class__.__name__ == "Image":
            return column_name
    for candidate in ("image", "img", "pixel_values"):
        if candidate in dataset.column_names:
            return candidate
    raise ValueError(f"Could not detect image column from columns={dataset.column_names}")


def detect_label_column_name(dataset) -> str:
    for column_name, feature in dataset.features.items():
        if feature.__class__.__name__ == "ClassLabel":
            return column_name
    for candidate in ("label", "labels", "category"):
        if candidate in dataset.column_names:
            return candidate
    raise ValueError(f"Could not detect label column from columns={dataset.column_names}")


def make_collate_fn(image_processor, image_column_name: str, label_column_name: str):
    def collate_fn(examples):
        images = [example[image_column_name].convert("RGB") for example in examples]
        labels = torch.tensor([int(example[label_column_name]) for example in examples], dtype=torch.long)
        inputs = image_processor(images=images, return_tensors="pt")
        inputs["labels"] = labels
        return inputs

    return collate_fn


@torch.no_grad()
def evaluate_model(model, dataloader, device: torch.device) -> Tuple[Dict[str, float], List[int], List[int]]:
    model.eval()
    losses: List[float] = []
    predictions: List[int] = []
    labels: List[int] = []

    for batch in dataloader:
        batch = {key: value.to(device) for key, value in batch.items()}
        outputs = model(**batch)
        losses.append(float(outputs.loss.detach().cpu()))
        logits = outputs.logits.detach().cpu()
        pred_batch = torch.argmax(logits, dim=-1).numpy().tolist()
        label_batch = batch["labels"].detach().cpu().numpy().tolist()
        predictions.extend(pred_batch)
        labels.extend(label_batch)

    accuracy = float(np.mean(np.asarray(predictions) == np.asarray(labels))) if labels else 0.0
    metrics = {
        "accuracy": accuracy,
        "primary_metric": "accuracy",
        "primary_metric_value": accuracy,
        "eval_loss": float(np.mean(losses)) if losses else 0.0,
    }
    model.train()
    return metrics, predictions, labels


def save_method_config(output_dir: Path, config: Dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "lora_config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True)


def save_checkpoint(model, image_processor, checkpoint_dir: Path, method_config: Dict[str, object]) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_trainable_state(model, checkpoint_dir / "trainable_state.pt")
    save_method_config(checkpoint_dir, method_config)
    image_processor.save_pretrained(checkpoint_dir)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    task_name = args.task_name
    metadata = get_task_metadata(task_name)
    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / "checkpoint"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    target_modules = tuple(part.strip() for part in args.target_modules.split(",") if part.strip())

    raw_dataset, image_processor, model, label_names, label_column_name = load_image_processor_and_model(
        args.model_name,
        task_name,
    )
    replaced_modules: List[str] = []
    if args.method == "lora":
        replaced_modules = inject_lora_into_vision_encoder(
            model,
            r=args.lora_r,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            target_modules=target_modules,
        )
        mark_only_lora_and_head_as_trainable(model)
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
            "dataset_name": metadata["dataset_name"],
            "train_split": metadata["train_split"],
            "eval_split": metadata["eval_split"],
            "device": str(device),
            "target_modules": list(target_modules),
            "label_names": label_names,
            "label_column_name": label_column_name,
            "replaced_modules": replaced_modules,
            "parameter_count": parameter_count,
        }
    )
    write_json(output_dir / "train_config.json", train_config)

    train_dataset = raw_dataset[str(metadata["train_split"])]
    eval_dataset = raw_dataset[str(metadata["eval_split"])]
    if args.max_train_samples is not None:
        train_dataset = train_dataset.select(range(min(args.max_train_samples, len(train_dataset))))
    if args.max_eval_samples is not None:
        eval_dataset = eval_dataset.select(range(min(args.max_eval_samples, len(eval_dataset))))

    image_column_name = detect_image_column_name(train_dataset)
    train_config["image_column_name"] = image_column_name
    write_json(output_dir / "train_config.json", train_config)

    collate_fn = make_collate_fn(image_processor, image_column_name, label_column_name)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

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
        "dataset_name": metadata["dataset_name"],
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "target_modules": list(target_modules),
        "num_labels": len(label_names),
        "replaced_modules": replaced_modules,
    }

    best_metric = -float("inf")
    best_metrics: Dict[str, float] = {}
    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    model.train()

    progress = tqdm(total=total_update_steps, desc=f"{task_name}-{args.method}")
    for epoch in range(1, args.epochs + 1):
        running_loss = 0.0
        for step, batch in enumerate(train_loader, start=1):
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / args.gradient_accumulation_steps
            loss.backward()
            running_loss += float(loss.detach().cpu())

            if step % args.gradient_accumulation_steps == 0 or step == len(train_loader):
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                progress.update(1)

        eval_metrics, predictions, labels = evaluate_model(model, eval_loader, device)
        epoch_log = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": running_loss / max(1, len(train_loader)),
            **eval_metrics,
        }
        append_jsonl(output_dir / "train_log.jsonl", epoch_log)

        if eval_metrics["primary_metric_value"] >= best_metric:
            best_metric = float(eval_metrics["primary_metric_value"])
            best_metrics = dict(eval_metrics)
            save_checkpoint(model, image_processor, checkpoint_dir, method_config)
            write_json(
                output_dir / "eval_predictions.json",
                {"predictions": predictions, "labels": labels},
            )

    progress.close()

    final_metrics = {
        **best_metrics,
        "task_name": task_name,
        "dataset_name": metadata["dataset_name"],
        "method": args.method,
        "model_name": args.model_name,
        "train_samples": len(train_dataset),
        "eval_samples": len(eval_dataset),
    }
    write_json(output_dir / "metrics.json", final_metrics)
    print(json.dumps(final_metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
