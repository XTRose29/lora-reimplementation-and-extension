"""Minimal LoRA layers used by our NLU reimplementation.

This file does not import the original Microsoft `loralib`. It reimplements
the core LoRA idea we need for encoder-based classification: freeze an existing
projection and add a trainable low-rank update.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Dict, Iterable, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
from torch import nn
import torch.nn.functional as F


class MyLoRALinear(nn.Module):
    """Wrap an existing nn.Linear with a LoRA low-rank update.

    Given a frozen base projection W, LoRA trains two smaller matrices A and B:

        y = x W^T + bias + (alpha / r) * x A^T B^T

    `lora_A` is initialized randomly and `lora_B` is initialized to zero, so the
    wrapped model starts from exactly the base model's behavior.
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        r: int = 4,
        lora_alpha: float = 32.0,
        lora_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError(f"MyLoRALinear expects nn.Linear, got {type(base_layer)!r}")
        if r <= 0:
            raise ValueError("LoRA rank r must be positive")

        self.base_layer = base_layer
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.r = int(r)
        self.lora_alpha = float(lora_alpha)
        self.scaling = self.lora_alpha / self.r
        self.lora_dropout = nn.Dropout(p=float(lora_dropout)) if lora_dropout > 0 else nn.Identity()

        device = base_layer.weight.device
        dtype = base_layer.weight.dtype
        self.lora_A = nn.Parameter(torch.empty(self.r, self.in_features, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.empty(self.out_features, self.r, device=device, dtype=dtype))
        self.reset_lora_parameters()

        for param in self.base_layer.parameters():
            param.requires_grad = False

    @classmethod
    def from_linear(
        cls,
        layer: nn.Linear,
        r: int,
        lora_alpha: float,
        lora_dropout: float,
    ) -> "MyLoRALinear":
        return cls(layer, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)

    def reset_lora_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_output = self.base_layer(x)
        lora_hidden = F.linear(self.lora_dropout(x), self.lora_A)
        lora_update = F.linear(lora_hidden, self.lora_B)
        return base_output + self.scaling * lora_update

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"r={self.r}, lora_alpha={self.lora_alpha}, scaling={self.scaling:.4f}"
        )


def mark_only_lora_and_head_as_trainable(
    model: nn.Module,
    head_keywords: Iterable[str] = ("classifier", "score", "pre_classifier"),
) -> None:
    """Freeze everything except LoRA parameters and the task head."""

    head_keywords = tuple(head_keywords)
    for name, param in model.named_parameters():
        is_lora = "lora_A" in name or "lora_B" in name
        is_head = any(keyword in name for keyword in head_keywords)
        param.requires_grad = is_lora or is_head


def mark_only_bias_and_head_as_trainable(
    model: nn.Module,
    head_keywords: Iterable[str] = ("classifier", "score", "pre_classifier"),
) -> None:
    """BitFit baseline: train only bias terms plus the task head."""

    head_keywords = tuple(head_keywords)
    for name, param in model.named_parameters():
        is_bias = name.endswith(".bias") or ".bias" in name
        is_head = any(keyword in name for keyword in head_keywords)
        param.requires_grad = is_bias or is_head


def mark_all_as_trainable(model: nn.Module) -> None:
    """Full fine-tuning baseline."""

    for param in model.parameters():
        param.requires_grad = True


def count_trainable_parameters(model: nn.Module) -> Dict[str, float]:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return {
        "total_parameters": int(total),
        "trainable_parameters": int(trainable),
        "frozen_parameters": int(total - trainable),
        "trainable_ratio": float(trainable / total) if total else 0.0,
    }


def trainable_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Return only trainable parameters, normally LoRA + classification head."""

    return {
        name: param.detach().cpu()
        for name, param in model.named_parameters()
        if param.requires_grad
    }


def save_trainable_state(model: nn.Module, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": trainable_state_dict(model)}, path)


def load_trainable_state(
    model: nn.Module,
    path: str | Path,
    map_location: str | torch.device = "cpu",
) -> Tuple[list, list]:
    checkpoint = torch.load(path, map_location=map_location)
    state_dict = checkpoint.get("state_dict", checkpoint)
    incompatible = model.load_state_dict(state_dict, strict=False)
    return list(incompatible.missing_keys), list(incompatible.unexpected_keys)
