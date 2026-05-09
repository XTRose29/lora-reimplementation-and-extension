"""Train/evaluate RoBERTa-base FT and LoRA reliability on GLUE tasks.

This script intentionally imports the local reimplementation:
`reimpl.my_lora.MyLoRALinear` is the LoRA layer used for all LoRA variants.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "reimpl") not in sys.path:
    sys.path.insert(0, str(ROOT / "reimpl"))

import numpy as np
import torch
from datasets import load_dataset
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
from transformers import get_linear_schedule_with_warmup

from reimpl.my_lora import MyLoRALinear, count_trainable_parameters, save_trainable_state
from reimpl.my_modeling import compute_task_metrics, get_task_metadata

try:
    from .metrics import classification_reliability_metrics, softmax
except ImportError:
    from metrics import classification_reliability_metrics, softmax


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GLUE calibration/abstention experiment.")
    parser.add_argument("--task_name", default="cola", choices=["cola", "mrpc", "rte", "sst2", "sst-2"])
    parser.add_argument("--model_name", default="roberta-base")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--method", choices=["ft", "lora"], default="lora")
    parser.add_argument("--lora_r", type=int, default=4)
    parser.add_argument("--lora_alpha", type=float, default=32.0)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--lora_placement", choices=["attention", "mlp", "attention_mlp"], default="attention")
    parser.add_argument("--attention_targets", default="auto")
    parser.add_argument("--layer_indices", default="all", help="all, comma list, or ranges like 0-5,9,11")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--max_ood_samples", type=int, default=1000)
    parser.add_argument("--ood_task", default="sst2", help="Confidence-only OOD split; use none to skip.")
    parser.add_argument("--ood_split", default="validation")
    parser.add_argument("--ood_has_compatible_labels", action="store_true")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--calibration_bins", type=int, default=15)
    parser.add_argument("--abstention_threshold", type=float, default=0.80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested_device: Optional[str]) -> torch.device:
    if requested_device:
        if requested_device.startswith("cuda"):
            ensure_cuda_supported()
        return torch.device(requested_device)
    if torch.cuda.is_available() and ensure_cuda_supported(raise_on_unsupported=False):
        return torch.device("cuda")
    return torch.device("cpu")


def ensure_cuda_supported(*, raise_on_unsupported: bool = True) -> bool:
    if not torch.cuda.is_available():
        if raise_on_unsupported:
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
        return False

    major, minor = torch.cuda.get_device_capability()
    device_sm = f"sm_{major}{minor}"
    supported_arches = set(torch.cuda.get_arch_list())
    if device_sm in supported_arches:
        return True

    message = (
        f"CUDA device capability {device_sm} is not supported by this PyTorch build. "
        f"Supported CUDA architectures: {', '.join(sorted(supported_arches)) or 'unknown'}. "
        "Use --device cpu, install a PyTorch build that supports this GPU, or run on a compatible GPU."
    )
    if raise_on_unsupported:
        raise RuntimeError(message)
    print(f"WARNING: {message} Falling back to CPU.", file=sys.stderr)
    return False


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def append_jsonl(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


def parse_layer_indices(value: str, num_layers: int) -> Optional[set[int]]:
    if value.lower() == "all":
        return None
    selected: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            selected.update(range(int(start), int(end) + 1))
        else:
            selected.add(int(part))
    invalid = [idx for idx in selected if idx < 0 or idx >= num_layers]
    if invalid:
        raise ValueError(f"Layer indices out of range 0-{num_layers - 1}: {invalid}")
    return selected


def get_parent_module(root: nn.Module, module_name: str) -> Tuple[nn.Module, str]:
    parts = module_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def transformer_layer_index(module_name: str) -> Optional[int]:
    match = re.search(r"\.(?:encoder\.layer|layers|h|block)\.(\d+)\.", f".{module_name}.")
    return int(match.group(1)) if match else None


def infer_attention_targets(model: nn.Module, requested_targets: str) -> Tuple[str, ...]:
    if requested_targets.lower() != "auto":
        return tuple(part.strip() for part in requested_targets.split(",") if part.strip())

    linear_leaf_names = {
        module_name.rsplit(".", 1)[-1]
        for module_name, module in model.named_modules()
        if isinstance(module, nn.Linear) and module_name
    }
    if {"query", "value"}.issubset(linear_leaf_names):
        return ("query", "value")
    if {"q_proj", "v_proj"}.issubset(linear_leaf_names):
        return ("q_proj", "v_proj")
    if {"c_attn"}.issubset(linear_leaf_names):
        return ("c_attn",)
    raise ValueError(
        "Could not infer attention LoRA targets. "
        f"Available linear leaf names include: {sorted(linear_leaf_names)[:30]}. "
        "Pass --attention_targets explicitly."
    )


def should_lora_wrap(
    module_name: str,
    module: nn.Module,
    *,
    placement: str,
    attention_targets: Sequence[str],
    layer_indices: Optional[set[int]],
) -> bool:
    if not isinstance(module, nn.Linear):
        return False
    layer_idx = transformer_layer_index(module_name)
    if layer_idx is None:
        return False
    if layer_indices is not None and layer_idx not in layer_indices:
        return False

    leaf = module_name.rsplit(".", 1)[-1]
    attention_markers = (
        ".attention.self.",
        ".self_attn.",
        ".attention.",
        ".attn.",
    )
    is_attention = any(marker in f".{module_name}." for marker in attention_markers) and leaf in set(attention_targets)
    is_mlp = module_name.endswith(".intermediate.dense") or (
        module_name.endswith(".output.dense") and ".attention." not in f".{module_name}."
    ) or leaf in {"gate_proj", "up_proj", "down_proj", "fc1", "fc2", "c_fc", "c_proj"}
    return (
        (placement == "attention" and is_attention)
        or (placement == "mlp" and is_mlp)
        or (placement == "attention_mlp" and (is_attention or is_mlp))
    )


def inject_lora(
    model: nn.Module,
    *,
    r: int,
    alpha: float,
    dropout: float,
    placement: str,
    attention_targets: Sequence[str],
    layer_indices: Optional[set[int]],
) -> List[str]:
    replaced: List[str] = []
    named_modules = list(model.named_modules())
    for module_name, module in named_modules:
        if not should_lora_wrap(
            module_name,
            module,
            placement=placement,
            attention_targets=attention_targets,
            layer_indices=layer_indices,
        ):
            continue
        parent, child_name = get_parent_module(model, module_name)
        setattr(parent, child_name, MyLoRALinear.from_linear(module, r, alpha, dropout))
        replaced.append(module_name)
    if not replaced:
        raise ValueError(f"No LoRA modules matched placement={placement}, targets={attention_targets}")
    return replaced


def mark_lora_and_head_trainable(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        is_lora = "lora_A" in name or "lora_B" in name
        is_head = "classifier" in name or "score" in name or "pre_classifier" in name
        param.requires_grad = is_lora or is_head


def mark_all_trainable(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = True


def preprocess_glue(dataset, tokenizer, task_name: str, max_length: int):
    metadata = get_task_metadata(task_name)
    fields = tuple(metadata["text_fields"])

    def tokenize_batch(batch):
        texts = [batch[field] for field in fields]
        tokenized = tokenizer(*texts, truncation=True, max_length=max_length)
        if "label" in batch:
            tokenized["labels"] = batch["label"]
        return tokenized

    return dataset.map(tokenize_batch, batched=True, remove_columns=dataset.column_names)


def preprocess_ood(dataset, tokenizer, task_name: str, max_length: int, keep_labels: bool):
    metadata = get_task_metadata(task_name)
    fields = tuple(metadata["text_fields"])

    def tokenize_batch(batch):
        texts = [batch[field] for field in fields]
        tokenized = tokenizer(*texts, truncation=True, max_length=max_length)
        if keep_labels and "label" in batch:
            tokenized["labels"] = batch["label"]
        return tokenized

    return dataset.map(tokenize_batch, batched=True, remove_columns=dataset.column_names)


@torch.no_grad()
def run_eval(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    *,
    task_name: str,
    labels_available: bool,
    calibration_bins: int,
    abstention_threshold: float,
) -> Tuple[Dict[str, float], List[int], List[Optional[int]], List[List[float]]]:
    model.eval()
    losses: List[float] = []
    predictions: List[int] = []
    labels: List[Optional[int]] = []
    logits_dump: List[List[float]] = []

    for batch in dataloader:
        batch = {key: value.to(device) for key, value in batch.items()}
        outputs = model(**batch)
        if labels_available and getattr(outputs, "loss", None) is not None:
            losses.append(float(outputs.loss.detach().cpu()))
        logits = outputs.logits.detach().cpu()
        logits_dump.extend(logits.numpy().tolist())
        predictions.extend(torch.argmax(logits, dim=-1).numpy().astype(int).tolist())
        if labels_available:
            labels.extend(batch["labels"].detach().cpu().numpy().astype(int).tolist())
        else:
            labels.extend([None] * logits.shape[0])

    logits_np = np.asarray(logits_dump, dtype=np.float64)
    label_np = None if not labels_available else np.asarray(labels, dtype=np.int64)
    reliability = classification_reliability_metrics(
        logits_np,
        label_np,
        n_bins=calibration_bins,
        abstention_threshold=abstention_threshold,
    )
    if labels_available:
        task_metrics = compute_task_metrics(task_name, predictions, label_np.tolist())
        reliability.update(task_metrics)
        reliability["eval_loss"] = float(np.mean(losses)) if losses else 0.0
    model.train()
    return reliability, predictions, labels, logits_dump


def save_predictions(path: Path, predictions: Sequence[int], labels: Sequence[Optional[int]], logits: Sequence[Sequence[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    probs = softmax(np.asarray(logits, dtype=np.float64))
    with path.open("w", encoding="utf-8") as f:
        for idx, (prediction, label, logit, prob) in enumerate(zip(predictions, labels, logits, probs.tolist())):
            f.write(
                json.dumps(
                    {
                        "idx": idx,
                        "prediction": int(prediction),
                        "label": None if label is None else int(label),
                        "confidence": float(max(prob)),
                        "logits": list(logit),
                        "probabilities": prob,
                    },
                    sort_keys=True,
                )
                + "\n"
            )


def main() -> None:
    args = parse_args()
    args.task_name = args.task_name.lower().replace("-", "")
    set_seed(args.seed)
    task_metadata = get_task_metadata(args.task_name)
    num_labels = int(task_metadata["num_labels"])
    validation_split = str(task_metadata["validation_split"])
    primary_metric = str(task_metadata["primary_metric"])

    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / "checkpoint"
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=num_labels)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    if tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    attention_targets = infer_attention_targets(model, args.attention_targets)
    num_layers = int(getattr(model.config, "num_hidden_layers", 12))
    selected_layers = parse_layer_indices(args.layer_indices, num_layers)

    replaced_modules: List[str] = []
    if args.method == "lora":
        replaced_modules = inject_lora(
            model,
            r=args.lora_r,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            placement=args.lora_placement,
            attention_targets=attention_targets,
            layer_indices=selected_layers,
        )
        mark_lora_and_head_trainable(model)
    else:
        mark_all_trainable(model)

    model.to(device)
    lr = args.learning_rate
    if lr is None:
        lr = 2e-5 if args.method == "ft" else 2e-4

    train_config = vars(args).copy()
    train_config.update(
        {
            "task_name": args.task_name,
            "validation_split": validation_split,
            "primary_metric": primary_metric,
            "device": str(device),
            "resolved_learning_rate": lr,
            "attention_targets": list(attention_targets),
            "selected_layers": "all" if selected_layers is None else sorted(selected_layers),
            "replaced_modules": replaced_modules,
            "parameter_count": count_trainable_parameters(model),
        }
    )
    write_json(output_dir / "train_config.json", train_config)
    write_json(output_dir / "parameter_count.json", train_config["parameter_count"])

    raw = load_dataset("glue", args.task_name)
    train_dataset = raw["train"]
    eval_dataset = raw[validation_split]
    if args.max_train_samples is not None:
        train_dataset = train_dataset.select(range(min(args.max_train_samples, len(train_dataset))))
    if args.max_eval_samples is not None:
        eval_dataset = eval_dataset.select(range(min(args.max_eval_samples, len(eval_dataset))))

    encoded_train = preprocess_glue(train_dataset, tokenizer, args.task_name, args.max_length)
    encoded_eval = preprocess_glue(eval_dataset, tokenizer, args.task_name, args.max_length)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_loader = DataLoader(encoded_train, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    eval_loader = DataLoader(encoded_eval, batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    optimizer = AdamW([param for param in model.parameters() if param.requires_grad], lr=lr, weight_decay=args.weight_decay)
    update_steps_per_epoch = max(1, int(np.ceil(len(train_loader) / args.gradient_accumulation_steps)))
    total_steps = max(1, args.epochs * update_steps_per_epoch)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.warmup_ratio * total_steps),
        num_training_steps=total_steps,
    )

    best_metric = -float("inf")
    best_metrics: Dict[str, float] = {}
    optimizer.zero_grad(set_to_none=True)
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: List[float] = []
        progress = tqdm(train_loader, desc=f"{output_dir.name} epoch {epoch}/{args.epochs}", leave=False)
        for step, batch in enumerate(progress, start=1):
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / args.gradient_accumulation_steps
            loss.backward()
            losses.append(float(outputs.loss.detach().cpu()))
            if step % args.gradient_accumulation_steps == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                progress.set_postfix(loss=f"{np.mean(losses[-20:]):.4f}")

        eval_metrics, _, _, _ = run_eval(
            model,
            eval_loader,
            device,
            task_name=args.task_name,
            labels_available=True,
            calibration_bins=args.calibration_bins,
            abstention_threshold=args.abstention_threshold,
        )
        record = {"epoch": epoch, "global_step": global_step, "train_loss": float(np.mean(losses)), **eval_metrics}
        append_jsonl(output_dir / "train_log.jsonl", record)
        current_metric = float(eval_metrics.get(primary_metric, 0.0))
        if current_metric > best_metric:
            best_metric = current_metric
            best_metrics = dict(eval_metrics)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            save_trainable_state(model, checkpoint_dir / "trainable_state.pt")
            tokenizer.save_pretrained(checkpoint_dir)
            write_json(checkpoint_dir / "experiment_config.json", train_config)

    id_metrics, id_predictions, id_labels, id_logits = run_eval(
        model,
        eval_loader,
        device,
        task_name=args.task_name,
        labels_available=True,
        calibration_bins=args.calibration_bins,
        abstention_threshold=args.abstention_threshold,
    )
    id_metrics.update(
        {
            "split": validation_split,
            "distribution": "id",
            f"best_{primary_metric}": best_metric,
            "best_primary_metric": best_metric,
        }
    )
    write_json(output_dir / "id_metrics.json", id_metrics)
    save_predictions(output_dir / "id_predictions.jsonl", id_predictions, id_labels, id_logits)

    final_metrics = {
        "method": args.method,
        "task_name": args.task_name,
        "model_name": args.model_name,
        **best_metrics,
        "final_id": id_metrics,
    }
    if args.ood_task.lower() != "none":
        ood_raw = load_dataset("glue", args.ood_task.lower())[args.ood_split]
        if args.max_ood_samples is not None:
            ood_raw = ood_raw.select(range(min(args.max_ood_samples, len(ood_raw))))
        encoded_ood = preprocess_ood(
            ood_raw,
            tokenizer,
            args.ood_task.lower(),
            args.max_length,
            keep_labels=args.ood_has_compatible_labels,
        )
        ood_loader = DataLoader(encoded_ood, batch_size=args.batch_size, shuffle=False, collate_fn=collator)
        ood_metrics, ood_predictions, ood_labels, ood_logits = run_eval(
            model,
            ood_loader,
            device,
            task_name=args.ood_task.lower(),
            labels_available=args.ood_has_compatible_labels,
            calibration_bins=args.calibration_bins,
            abstention_threshold=args.abstention_threshold,
        )
        ood_metrics.update({"split": args.ood_split, "distribution": "ood", "ood_task": args.ood_task})
        write_json(output_dir / "ood_metrics.json", ood_metrics)
        save_predictions(output_dir / "ood_predictions.jsonl", ood_predictions, ood_labels, ood_logits)
        final_metrics["ood"] = ood_metrics

    write_json(output_dir / "metrics.json", final_metrics)
    print(f"Done: {output_dir}")
    print(
        f"ID {primary_metric}={id_metrics.get(primary_metric, 0.0):.4f} "
        f"ECE={id_metrics.get('ece', 0.0):.4f} "
        f"abstain={id_metrics.get('abstention_rate', 0.0):.4f}"
    )


if __name__ == "__main__":
    main()
