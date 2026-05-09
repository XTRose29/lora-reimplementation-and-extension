"""Probe utilities for task-sensitive LoRA placement."""

from __future__ import annotations

import json
import math
import random
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from torch import nn

from reimpl.my_lora import MyLoRALinear


ATTENTION_MARKERS = (".attention.self.", ".self_attn.", ".attention.", ".attn.")


def write_json(path: str | Path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def read_json(path: str | Path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def get_parent_module(root: nn.Module, module_name: str) -> Tuple[nn.Module, str]:
    parts = module_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def _is_attention_candidate(module_name: str, module: nn.Module, target_modules: Sequence[str]) -> bool:
    if not isinstance(module, nn.Linear):
        return False
    leaf_name = module_name.rsplit(".", 1)[-1]
    if leaf_name not in set(target_modules):
        return False
    return any(marker in f".{module_name}." for marker in ATTENTION_MARKERS)


def _is_mlp_candidate(module_name: str, module: nn.Module) -> bool:
    if not isinstance(module, nn.Linear):
        return False
    return module_name.endswith(".intermediate.dense") or (
        module_name.endswith(".output.dense") and ".attention." not in module_name
    )


def find_candidate_linear_modules(
    model: nn.Module,
    target_modules: Iterable[str] = ("query", "value"),
    candidate_scope: str = "attention",
) -> List[str]:
    """Return exact module names that may receive LoRA."""

    if candidate_scope not in {"attention", "attention_mlp"}:
        raise ValueError("candidate_scope must be one of: attention, attention_mlp")

    target_modules = tuple(target_modules)
    candidates: List[str] = []
    for module_name, module in model.named_modules():
        if not module_name:
            continue
        is_attention = _is_attention_candidate(module_name, module, target_modules)
        is_mlp = candidate_scope == "attention_mlp" and _is_mlp_candidate(module_name, module)
        if is_attention or is_mlp:
            candidates.append(module_name)
    if not candidates:
        examples = [name for name, module in model.named_modules() if isinstance(module, nn.Linear)][:20]
        raise ValueError(f"No LoRA candidate modules found. Linear examples={examples}")
    return candidates


def inject_lora_into_selected_modules(
    model: nn.Module,
    selected_module_names: Sequence[str],
    r: int = 4,
    alpha: float = 32.0,
    dropout: float = 0.1,
) -> List[str]:
    """Replace exact selected nn.Linear modules with MyLoRALinear wrappers."""

    selected = set(selected_module_names)
    replaced: List[str] = []
    named_modules = dict(model.named_modules())
    missing = sorted(name for name in selected if name not in named_modules)
    if missing:
        raise ValueError(f"Selected modules not found in model: {missing[:10]}")

    for module_name in selected_module_names:
        module = named_modules[module_name]
        if not isinstance(module, nn.Linear):
            raise TypeError(f"Selected module is not nn.Linear: {module_name} ({type(module)!r})")
        parent, child_name = get_parent_module(model, module_name)
        setattr(parent, child_name, MyLoRALinear.from_linear(module, r, alpha, dropout))
        replaced.append(module_name)
    return replaced


def _rms_norm(tensor: torch.Tensor) -> float:
    value = tensor.detach().float()
    if value.numel() == 0:
        return 0.0
    return float(torch.sqrt(torch.mean(value * value)).cpu())


def _set_only_candidates_require_grad(model: nn.Module, candidate_names: Sequence[str]) -> Dict[str, bool]:
    original_requires_grad = {name: param.requires_grad for name, param in model.named_parameters()}
    for param in model.parameters():
        param.requires_grad = False
    named_modules = dict(model.named_modules())
    for module_name in candidate_names:
        named_modules[module_name].weight.requires_grad = True
    return original_requires_grad


def _restore_requires_grad(model: nn.Module, original_requires_grad: Dict[str, bool]) -> None:
    for name, param in model.named_parameters():
        if name in original_requires_grad:
            param.requires_grad = original_requires_grad[name]


def _register_activation_hooks(
    model: nn.Module,
    candidate_names: Sequence[str],
    activation_sums: Dict[str, float],
    activation_counts: Dict[str, int],
):
    handles = []
    named_modules = dict(model.named_modules())

    def make_hook(name: str):
        def hook(_module, _inputs, output):
            if isinstance(output, tuple):
                output = output[0]
            activation_sums[name] += _rms_norm(output)
            activation_counts[name] += 1

        return hook

    for module_name in candidate_names:
        handles.append(named_modules[module_name].register_forward_hook(make_hook(module_name)))
    return handles


def collect_probe_scores(
    model: nn.Module,
    dataloader,
    candidate_names: Sequence[str],
    device: torch.device,
    max_probe_batches: int | None = None,
) -> List[Dict[str, float | str | int]]:
    """Collect activation and gradient energy for candidate modules."""

    model.to(device)
    model.eval()
    activation_sums = {name: 0.0 for name in candidate_names}
    activation_counts = {name: 0 for name in candidate_names}
    gradient_sums = {name: 0.0 for name in candidate_names}
    gradient_counts = {name: 0 for name in candidate_names}

    original_requires_grad = _set_only_candidates_require_grad(model, candidate_names)
    handles = _register_activation_hooks(model, candidate_names, activation_sums, activation_counts)
    named_modules = dict(model.named_modules())

    try:
        for batch_idx, batch in enumerate(dataloader, start=1):
            if max_probe_batches is not None and batch_idx > max_probe_batches:
                break
            model.zero_grad(set_to_none=True)
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            outputs.loss.backward()
            for module_name in candidate_names:
                grad = named_modules[module_name].weight.grad
                if grad is not None:
                    gradient_sums[module_name] += _rms_norm(grad)
                    gradient_counts[module_name] += 1
        model.zero_grad(set_to_none=True)
    finally:
        for handle in handles:
            handle.remove()
        _restore_requires_grad(model, original_requires_grad)

    rows: List[Dict[str, float | str | int]] = []
    for module_name in candidate_names:
        activation = activation_sums[module_name] / max(1, activation_counts[module_name])
        gradient = gradient_sums[module_name] / max(1, gradient_counts[module_name])
        rows.append(
            {
                "module_name": module_name,
                "activation_energy": float(activation),
                "gradient_energy": float(gradient),
                "activation_count": int(activation_counts[module_name]),
                "gradient_count": int(gradient_counts[module_name]),
            }
        )
    return add_normalized_scores(rows)


def add_normalized_scores(rows: List[Dict[str, float | str | int]]) -> List[Dict[str, float | str | int]]:
    max_activation = max((float(row["activation_energy"]) for row in rows), default=0.0)
    max_gradient = max((float(row["gradient_energy"]) for row in rows), default=0.0)
    eps = 1e-12
    for row in rows:
        activation_norm = float(row["activation_energy"]) / (max_activation + eps)
        gradient_norm = float(row["gradient_energy"]) / (max_gradient + eps)
        row["activation_norm"] = activation_norm
        row["gradient_norm"] = gradient_norm
        row["actgrad_score"] = math.sqrt(max(0.0, activation_norm * gradient_norm))
    return rows


def make_unprobed_score_rows(candidate_names: Sequence[str]) -> List[Dict[str, float | str | int]]:
    rows = []
    for module_name in candidate_names:
        rows.append(
            {
                "module_name": module_name,
                "activation_energy": 0.0,
                "gradient_energy": 0.0,
                "activation_count": 0,
                "gradient_count": 0,
                "activation_norm": 0.0,
                "gradient_norm": 0.0,
                "actgrad_score": 0.0,
            }
        )
    return rows


def select_modules_from_scores(
    rows: Sequence[Dict[str, float | str | int]],
    strategy: str,
    top_k: int | None,
    seed: int = 42,
) -> List[str]:
    """Select module names from score rows."""

    strategy = strategy.lower()
    module_names = [str(row["module_name"]) for row in rows]
    if strategy == "full":
        return module_names
    if top_k is None or top_k <= 0:
        raise ValueError("top_k must be positive unless strategy='full'")
    top_k = min(top_k, len(module_names))

    if strategy == "random":
        rng = random.Random(seed)
        selected = list(module_names)
        rng.shuffle(selected)
        return selected[:top_k]

    score_key_by_strategy = {
        "gradient": "gradient_energy",
        "activation": "activation_energy",
        "actgrad": "actgrad_score",
    }
    if strategy not in score_key_by_strategy:
        raise ValueError("strategy must be one of: full, random, gradient, activation, actgrad")

    score_key = score_key_by_strategy[strategy]
    sorted_rows = sorted(
        rows,
        key=lambda row: (float(row.get(score_key, 0.0)), str(row["module_name"])),
        reverse=True,
    )
    return [str(row["module_name"]) for row in sorted_rows[:top_k]]


def parse_roberta_layer_and_kind(module_name: str) -> Tuple[int | None, str]:
    layer_match = re.search(r"\.layer\.(\d+)\.", f".{module_name}.")
    layer = int(layer_match.group(1)) if layer_match else None
    if module_name.endswith(".query"):
        kind = "query"
    elif module_name.endswith(".value"):
        kind = "value"
    elif module_name.endswith(".key"):
        kind = "key"
    elif module_name.endswith(".intermediate.dense"):
        kind = "mlp_intermediate"
    elif module_name.endswith(".output.dense"):
        kind = "mlp_output"
    else:
        kind = module_name.rsplit(".", 1)[-1]
    return layer, kind


def annotate_scores_with_selection(
    rows: Sequence[Dict[str, float | str | int]],
    selected_modules: Sequence[str],
) -> List[Dict[str, float | str | int | bool | None]]:
    selected = set(selected_modules)
    annotated = []
    for row in rows:
        copied = dict(row)
        layer, kind = parse_roberta_layer_and_kind(str(row["module_name"]))
        copied["layer"] = layer
        copied["kind"] = kind
        copied["selected"] = str(row["module_name"]) in selected
        annotated.append(copied)
    return annotated
