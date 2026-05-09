"""Evaluate a saved vision experiment checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
from datasets import load_dataset
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoModelForImageClassification


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reimpl.my_lora import MyLoRALinear, load_trainable_state  # noqa: E402


TASK_METADATA: Dict[str, Dict[str, object]] = {
    "cifar10": {"dataset_name": "cifar10", "eval_split": "test"},
    "beans": {"dataset_name": "beans", "eval_split": "validation"},
    "oxford_iiit_pet": {"dataset_name": "timm/oxford-iiit-pet", "eval_split": "test"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a custom LoRA vision checkpoint.")
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


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
    r: int,
    alpha: float,
    dropout: float,
    target_modules: Iterable[str],
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
    return replaced


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
def evaluate_model(model, dataloader, device: torch.device):
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
    return {
        "accuracy": accuracy,
        "primary_metric": "accuracy",
        "primary_metric_value": accuracy,
        "eval_loss": float(np.mean(losses)) if losses else 0.0,
    }, predictions, labels


def main() -> None:
    args = parse_args()
    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with (checkpoint_dir / "lora_config.json").open("r", encoding="utf-8") as f:
        config = json.load(f)

    task_name = str(config["task_name"])
    metadata = TASK_METADATA[task_name]
    raw_dataset = load_dataset(str(metadata["dataset_name"]))
    eval_dataset = raw_dataset[str(metadata["eval_split"])]
    if args.max_eval_samples is not None:
        eval_dataset = eval_dataset.select(range(min(args.max_eval_samples, len(eval_dataset))))

    label_column_name = str(config.get("label_column_name") or detect_label_column_name(eval_dataset))
    label_names = list(eval_dataset.features[label_column_name].names)
    image_processor = AutoImageProcessor.from_pretrained(checkpoint_dir, use_fast=True)
    model = AutoModelForImageClassification.from_pretrained(
        str(config["model_name"]),
        num_labels=int(config["num_labels"]),
        id2label={i: name for i, name in enumerate(label_names)},
        label2id={name: i for i, name in enumerate(label_names)},
        ignore_mismatched_sizes=True,
    )

    if config["method"] == "lora":
        inject_lora_into_vision_encoder(
            model,
            r=int(config["lora_r"]),
            alpha=float(config["lora_alpha"]),
            dropout=float(config["lora_dropout"]),
            target_modules=tuple(config["target_modules"]),
        )

    missing_keys, unexpected_keys = load_trainable_state(model, checkpoint_dir / "trainable_state.pt")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)

    image_column_name = detect_image_column_name(eval_dataset)
    collate_fn = make_collate_fn(image_processor, image_column_name, label_column_name)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    metrics, predictions, labels = evaluate_model(model, eval_loader, device)
    metrics.update(
        {
            "task_name": task_name,
            "dataset_name": metadata["dataset_name"],
            "method": config["method"],
            "model_name": config["model_name"],
            "missing_keys": missing_keys,
            "unexpected_keys": unexpected_keys,
            "eval_samples": len(eval_dataset),
        }
    )

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    with (output_dir / "eval_predictions.json").open("w", encoding="utf-8") as f:
        json.dump({"predictions": predictions, "labels": labels}, f, indent=2, sort_keys=True)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
