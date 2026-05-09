"""Reliability metrics for CoLA calibration and abstention experiments."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def classification_reliability_metrics(
    logits: np.ndarray,
    labels: Optional[np.ndarray] = None,
    *,
    n_bins: int = 15,
    abstention_threshold: float = 0.80,
) -> Dict[str, float]:
    """Compute confidence, calibration, and selective-answering metrics.

    Calibration follows standard confidence calibration for classification:
    examples are binned by max softmax confidence, and ECE is the weighted
    average absolute difference between bin accuracy and bin confidence.
    """

    probs = softmax(logits)
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    attempted = confidences >= float(abstention_threshold)

    metrics: Dict[str, float] = {
        "mean_confidence": float(np.mean(confidences)) if len(confidences) else 0.0,
        "abstention_threshold": float(abstention_threshold),
        "abstention_rate": float(np.mean(~attempted)) if len(attempted) else 0.0,
        "coverage": float(np.mean(attempted)) if len(attempted) else 0.0,
    }

    if labels is None:
        return metrics

    labels = np.asarray(labels, dtype=np.int64)
    correct = predictions == labels
    metrics["accuracy"] = float(np.mean(correct)) if len(correct) else 0.0
    metrics["nll"] = negative_log_likelihood(probs, labels)
    metrics["brier"] = brier_score(probs, labels)

    ece_payload = expected_calibration_error(confidences, correct, n_bins=n_bins)
    metrics.update(ece_payload)
    metrics["calibration_accuracy"] = float(1.0 - metrics["ece"])

    if np.any(attempted):
        metrics["selective_accuracy"] = float(np.mean(correct[attempted]))
        metrics["selective_risk"] = float(1.0 - metrics["selective_accuracy"])
    else:
        metrics["selective_accuracy"] = 0.0
        metrics["selective_risk"] = 0.0

    if np.any(~attempted):
        metrics["abstained_error_rate"] = float(np.mean(~correct[~attempted]))
    else:
        metrics["abstained_error_rate"] = 0.0

    return metrics


def expected_calibration_error(
    confidences: np.ndarray,
    correct: np.ndarray,
    *,
    n_bins: int = 15,
) -> Dict[str, float]:
    confidences = np.asarray(confidences, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    mce = 0.0

    for bin_idx in range(n_bins):
        left = bin_edges[bin_idx]
        right = bin_edges[bin_idx + 1]
        if bin_idx == n_bins - 1:
            in_bin = (confidences >= left) & (confidences <= right)
        else:
            in_bin = (confidences >= left) & (confidences < right)
        if not np.any(in_bin):
            continue
        bin_acc = float(np.mean(correct[in_bin]))
        bin_conf = float(np.mean(confidences[in_bin]))
        gap = abs(bin_acc - bin_conf)
        ece += float(np.mean(in_bin)) * gap
        mce = max(mce, gap)

    return {"ece": float(ece), "mce": float(mce), "calibration_bins": int(n_bins)}


def negative_log_likelihood(probs: np.ndarray, labels: np.ndarray) -> float:
    eps = 1e-12
    clipped = np.clip(probs, eps, 1.0)
    return float(-np.mean(np.log(clipped[np.arange(len(labels)), labels])))


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(labels)), labels] = 1.0
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))
