"""Evaluate a saved audio experiment checkpoint."""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import torchaudio
from datasets import Audio, concatenate_datasets, load_dataset
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reimpl.my_lora import MyLoRALinear, load_trainable_state  # noqa: E402


TASK_METADATA: Dict[str, Dict[str, object]] = {
    "speech_commands": {
        "dataset_name": "google/speech_commands",
        "dataset_config": "v0.02",
        "trust_remote_code": True,
        "split_strategy": "named_splits",
        "train_split": "train",
        "eval_split": "validation",
    },
    "minds14_en": {
        "dataset_name": "PolyAI/minds14",
        "dataset_config": "en-US",
        "trust_remote_code": False,
        "split_strategy": "stratified_holdout",
        "train_split": "train",
        "eval_fraction": 0.2,
    },
    "superb_er": {
        "dataset_name": "s3prl/superb",
        "dataset_config": "er",
        "trust_remote_code": True,
        "split_strategy": "session_heldout",
        "train_splits": ["session1", "session2", "session3", "session4"],
        "eval_split": "session5",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a custom LoRA audio checkpoint.")
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_duration_seconds", type=float, default=8.0)
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
    attention_markers = (".attention.", ".attn.", ".encoder.layers.")
    return any(marker in f".{module_name}." for marker in attention_markers)


def inject_lora_into_audio_encoder(
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


def detect_audio_column_name(dataset) -> str:
    for column_name, feature in dataset.features.items():
        if feature.__class__.__name__ == "Audio":
            return column_name
    for candidate in ("audio", "speech"):
        if candidate in dataset.column_names:
            return candidate
    raise ValueError(f"Could not detect audio column from columns={dataset.column_names}")


def detect_label_column_name(dataset) -> str:
    for column_name, feature in dataset.features.items():
        if feature.__class__.__name__ == "ClassLabel":
            return column_name
    for candidate in ("label", "labels", "intent_class", "language"):
        if candidate in dataset.column_names:
            return candidate
    raise ValueError(f"Could not detect label column from columns={dataset.column_names}")


def load_audio_task(task_name: str, seed: int):
    metadata = TASK_METADATA[task_name]
    dataset_name = str(metadata["dataset_name"])
    dataset_config = metadata.get("dataset_config")
    trust_remote_code = bool(metadata.get("trust_remote_code", False))
    raw_dataset = (
        load_dataset(dataset_name, dataset_config, trust_remote_code=trust_remote_code)
        if dataset_config
        else load_dataset(dataset_name, trust_remote_code=trust_remote_code)
    )

    split_strategy = str(metadata["split_strategy"])
    if split_strategy == "named_splits":
        eval_dataset = raw_dataset[str(metadata["eval_split"])]
    elif split_strategy == "stratified_holdout":
        base_dataset = raw_dataset[str(metadata["train_split"])]
        label_column_name = detect_label_column_name(base_dataset)
        split_dataset = base_dataset.train_test_split(
            test_size=float(metadata["eval_fraction"]),
            seed=seed,
            stratify_by_column=label_column_name,
        )
        eval_dataset = split_dataset["test"]
    elif split_strategy == "session_heldout":
        eval_dataset = raw_dataset[str(metadata["eval_split"])]
    else:
        raise ValueError(f"Unsupported split strategy: {split_strategy}")

    return eval_dataset, metadata


def load_audio_array(audio_value: Dict[str, object], target_sampling_rate: int) -> np.ndarray:
    if audio_value.get("array") is not None:
        array = np.asarray(audio_value["array"], dtype=np.float32)
        sampling_rate = int(audio_value["sampling_rate"])
        waveform = torch.from_numpy(array).unsqueeze(0)
    else:
        path = audio_value.get("path")
        audio_bytes = audio_value.get("bytes")
        if path:
            waveform, sampling_rate = torchaudio.load(path)
        elif audio_bytes is not None:
            waveform, sampling_rate = torchaudio.load(io.BytesIO(audio_bytes))
        else:
            raise ValueError("Audio value must contain decoded audio, a file path, or raw bytes.")

    if waveform.ndim == 2 and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sampling_rate != target_sampling_rate:
        waveform = torchaudio.functional.resample(waveform, sampling_rate, target_sampling_rate)
    return waveform.squeeze(0).numpy()


def make_collate_fn(feature_extractor, audio_column_name: str, label_column_name: str, max_duration_seconds: float):
    def collate_fn(examples):
        audio_arrays = []
        labels = []
        sampling_rate = int(feature_extractor.sampling_rate)
        max_length = int(max_duration_seconds * sampling_rate)

        for example in examples:
            array = load_audio_array(example[audio_column_name], sampling_rate)
            if max_length > 0 and len(array) > max_length:
                array = array[:max_length]
            audio_arrays.append(array)
            labels.append(int(example[label_column_name]))

        inputs = feature_extractor(
            audio_arrays,
            sampling_rate=sampling_rate,
            return_tensors="pt",
            padding=True,
        )
        inputs["labels"] = torch.tensor(labels, dtype=torch.long)
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
    seed = int(config.get("seed", 42))
    eval_dataset, metadata = load_audio_task(task_name, seed)
    if args.max_eval_samples is not None:
        eval_dataset = eval_dataset.select(range(min(args.max_eval_samples, len(eval_dataset))))

    audio_column_name = str(config.get("audio_column_name") or detect_audio_column_name(eval_dataset))
    label_column_name = str(config.get("label_column_name") or detect_label_column_name(eval_dataset))

    feature_extractor = AutoFeatureExtractor.from_pretrained(checkpoint_dir)
    eval_dataset = eval_dataset.cast_column(audio_column_name, Audio(sampling_rate=feature_extractor.sampling_rate))
    label_names = list(eval_dataset.features[label_column_name].names)
    model = AutoModelForAudioClassification.from_pretrained(
        str(config["model_name"]),
        num_labels=int(config["num_labels"]),
        id2label={i: name for i, name in enumerate(label_names)},
        label2id={name: i for i, name in enumerate(label_names)},
        ignore_mismatched_sizes=True,
    )

    if config["method"] == "lora":
        inject_lora_into_audio_encoder(
            model,
            r=int(config["lora_r"]),
            alpha=float(config["lora_alpha"]),
            dropout=float(config["lora_dropout"]),
            target_modules=tuple(config["target_modules"]),
        )

    missing_keys, unexpected_keys = load_trainable_state(model, checkpoint_dir / "trainable_state.pt")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)

    collate_fn = make_collate_fn(
        feature_extractor,
        audio_column_name=audio_column_name,
        label_column_name=label_column_name,
        max_duration_seconds=args.max_duration_seconds,
    )
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    metrics, predictions, labels = evaluate_model(model, eval_loader, device)
    metrics.update(
        {
            "task_name": task_name,
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
