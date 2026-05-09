"""Model loading, LoRA injection, and GLUE metric utilities."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
from torch import nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer

try:
    from .my_lora import MyLoRALinear
    from .my_adapter import ModuleWithAdapter
except ImportError:
    from my_lora import MyLoRALinear
    from my_adapter import ModuleWithAdapter


TASK_METADATA: Dict[str, Dict[str, object]] = {
    "sst2": {
        "num_labels": 2,
        "text_fields": ("sentence",),
        "validation_split": "validation",
        "primary_metric": "accuracy",
    },
    "mrpc": {
        "num_labels": 2,
        "text_fields": ("sentence1", "sentence2"),
        "validation_split": "validation",
        "primary_metric": "f1",
    },
    "rte": {
        "num_labels": 2,
        "text_fields": ("sentence1", "sentence2"),
        "validation_split": "validation",
        "primary_metric": "accuracy",
    },
    "qnli": {
        "num_labels": 2,
        "text_fields": ("question", "sentence"),
        "validation_split": "validation",
        "primary_metric": "accuracy",
    },
    "qqp": {
        "num_labels": 2,
        "text_fields": ("question1", "question2"),
        "validation_split": "validation",
        "primary_metric": "f1",
    },
    "cola": {
        "num_labels": 2,
        "text_fields": ("sentence",),
        "validation_split": "validation",
        "primary_metric": "matthews_correlation",
    },
    "mnli": {
        "num_labels": 3,
        "text_fields": ("premise", "hypothesis"),
        "validation_split": "validation_matched",
        "primary_metric": "accuracy",
    },
    "stsb": {
        "num_labels": 1,
        "text_fields": ("sentence1", "sentence2"),
        "validation_split": "validation",
        "primary_metric": "pearson",
    },
}


def get_task_metadata(task_name: str) -> Dict[str, object]:
    task_name = task_name.lower()
    if task_name not in TASK_METADATA:
        supported = ", ".join(sorted(TASK_METADATA))
        raise ValueError(f"Unsupported GLUE task '{task_name}'. Supported tasks: {supported}")
    return TASK_METADATA[task_name]


def load_tokenizer_and_sequence_classifier(
    model_name: str,
    task_name: str,
    num_labels: int | None = None,
):
    metadata = get_task_metadata(task_name)
    num_labels = int(num_labels or metadata["num_labels"])
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
        model.config.pad_token_id = tokenizer.pad_token_id
    if task_name.lower() == "stsb":
        model.config.problem_type = "regression"
    return tokenizer, model


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
    attention_markers = (".attention.self.", ".self_attn.", ".attention.", ".attn.")
    return any(marker in f".{module_name}." for marker in attention_markers)


def inject_lora_into_encoder(
    model: nn.Module,
    r: int = 4,
    alpha: float = 32.0,
    dropout: float = 0.1,
    target_modules: Iterable[str] = ("query", "value"),
) -> List[str]:
    """Replace attention target nn.Linear modules with MyLoRALinear wrappers."""

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


def inject_adapter_into_encoder(
    model: nn.Module,
    adapter_size: int = 16,
    dropout: float = 0.0,
    location: str = "output",
) -> List[str]:
    """Wrap transformer output submodules with a small residual adapter.

    `location="output"` inserts one adapter after each transformer layer output.
    This gives roughly 2 * hidden_size * adapter_size * num_layers trainable
    adapter weights, so RoBERTa-base with adapter_size=16 is close to 0.3M.
    """

    if location not in {"output", "attention_output", "both"}:
        raise ValueError("adapter location must be one of: output, attention_output, both")

    hidden_size = int(getattr(model.config, "hidden_size", 0) or getattr(model.config, "dim", 0))
    if hidden_size <= 0:
        raise ValueError("Could not infer hidden size from model.config")

    replaced: List[str] = []
    named_modules = list(model.named_modules())
    for module_name, module in named_modules:
        if not module_name:
            continue
        is_attention_output = module_name.endswith(".attention.output")
        is_layer_output = module_name.endswith(".output") and ".attention." not in module_name
        should_wrap = (
            (location in {"attention_output", "both"} and is_attention_output)
            or (location in {"output", "both"} and is_layer_output)
        )
        if not should_wrap:
            continue
        parent, child_name = _get_parent_module(model, module_name)
        if isinstance(getattr(parent, child_name), ModuleWithAdapter):
            continue
        setattr(parent, child_name, ModuleWithAdapter(module, hidden_size, adapter_size, dropout))
        replaced.append(module_name)

    if not replaced:
        examples = [name for name, _ in named_modules if name.endswith(".output")][:10]
        raise ValueError(f"No adapter locations were replaced. Example output modules={examples}")
    return replaced


def save_lora_config(output_dir: str | Path, config: Dict[str, object]) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "lora_config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True)


def load_lora_config(adapter_dir: str | Path) -> Dict[str, object]:
    adapter_dir = Path(adapter_dir)
    with (adapter_dir / "lora_config.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def compute_task_metrics(task_name: str, predictions: Sequence[float], labels: Sequence[float]) -> Dict[str, float | str]:
    task_name = task_name.lower()
    metadata = get_task_metadata(task_name)
    primary_metric = str(metadata["primary_metric"])
    preds = np.asarray(predictions)
    refs = np.asarray(labels)
    metrics: Dict[str, float | str] = {}

    if task_name == "stsb":
        preds = preds.astype(float)
        refs = refs.astype(float)
        mse = float(np.mean((preds - refs) ** 2))
        pearson = float(np.corrcoef(preds, refs)[0, 1]) if len(preds) > 1 else 0.0
        metrics.update({"mse": mse, "pearson": 0.0 if np.isnan(pearson) else pearson})
    else:
        preds = preds.astype(int)
        refs = refs.astype(int)
        metrics["accuracy"] = float(np.mean(preds == refs))
        if task_name in {"mrpc", "qqp"}:
            metrics["f1"] = _binary_f1(refs, preds)
        if task_name == "cola":
            metrics["matthews_correlation"] = _binary_mcc(refs, preds)

    metrics["primary_metric"] = primary_metric
    metrics["primary_metric_value"] = float(metrics.get(primary_metric, 0.0))
    return metrics


def _binary_f1(labels: np.ndarray, predictions: np.ndarray) -> float:
    tp = float(np.sum((labels == 1) & (predictions == 1)))
    fp = float(np.sum((labels == 0) & (predictions == 1)))
    fn = float(np.sum((labels == 1) & (predictions == 0)))
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0


def _binary_mcc(labels: np.ndarray, predictions: np.ndarray) -> float:
    tp = float(np.sum((labels == 1) & (predictions == 1)))
    tn = float(np.sum((labels == 0) & (predictions == 0)))
    fp = float(np.sum((labels == 0) & (predictions == 1)))
    fn = float(np.sum((labels == 1) & (predictions == 0)))
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return float(((tp * tn) - (fp * fn)) / denominator) if denominator > 0 else 0.0
