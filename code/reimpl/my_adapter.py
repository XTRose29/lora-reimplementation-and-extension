"""Small bottleneck adapters for baseline comparisons.

The project focus is LoRA, but the paper table also includes adapter baselines.
This file implements a compact residual bottleneck adapter so we can run a
rough comparison with the same training script.
"""

from __future__ import annotations

from typing import Iterable, List

import torch
from torch import nn


class MyBottleneckAdapter(nn.Module):
    """A simple residual adapter: x + up(activation(down(x)))."""

    def __init__(self, hidden_size: int, adapter_size: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.adapter_down = nn.Linear(hidden_size, adapter_size)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.adapter_up = nn.Linear(adapter_size, hidden_size)
        nn.init.normal_(self.adapter_down.weight, std=0.02)
        nn.init.zeros_(self.adapter_down.bias)
        nn.init.zeros_(self.adapter_up.weight)
        nn.init.zeros_(self.adapter_up.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.adapter_down(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.adapter_up(hidden_states)
        return residual + hidden_states


class ModuleWithAdapter(nn.Module):
    """Wrap a transformer submodule and apply an adapter to its tensor output."""

    def __init__(self, base_module: nn.Module, hidden_size: int, adapter_size: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.base_module = base_module
        self.adapter = MyBottleneckAdapter(hidden_size, adapter_size, dropout)

    def forward(self, *args, **kwargs):
        output = self.base_module(*args, **kwargs)
        if isinstance(output, tuple):
            first = self.adapter(output[0])
            return (first, *output[1:])
        return self.adapter(output)


def mark_only_adapter_and_head_as_trainable(
    model: nn.Module,
    head_keywords: Iterable[str] = ("classifier", "score", "pre_classifier"),
) -> None:
    head_keywords = tuple(head_keywords)
    for name, param in model.named_parameters():
        is_adapter = ".adapter." in name or name.startswith("adapter.")
        is_head = any(keyword in name for keyword in head_keywords)
        param.requires_grad = is_adapter or is_head


def adapter_state_dict(model: nn.Module):
    return {
        name: param.detach().cpu()
        for name, param in model.named_parameters()
        if param.requires_grad
    }
