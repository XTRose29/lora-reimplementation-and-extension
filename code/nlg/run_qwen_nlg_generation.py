from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.request import urlopen

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "reimpl") not in sys.path:
    sys.path.insert(0, str(ROOT / "reimpl"))

import numpy as np
import pandas as pd
import torch
from datasets import Dataset as HFDataset
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset as TorchDataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from transformers.pytorch_utils import Conv1D

from reimpl.my_lora import MyLoRALinear, count_trainable_parameters


class MyLoRAConv1D(nn.Module):
    """LoRA wrapper for GPT-2's merged Conv1D attention projection."""

    def __init__(self, base_layer: Conv1D, r: int = 4, lora_alpha: float = 32.0, lora_dropout: float = 0.0) -> None:
        super().__init__()
        if r <= 0:
            raise ValueError("LoRA rank r must be positive")
        self.base_layer = base_layer
        self.in_features = int(base_layer.weight.shape[0])
        self.out_features = int(base_layer.weight.shape[1])
        self.r = int(r)
        self.lora_alpha = float(lora_alpha)
        self.scaling = self.lora_alpha / self.r
        self.lora_dropout = nn.Dropout(p=float(lora_dropout)) if lora_dropout > 0 else nn.Identity()

        device = base_layer.weight.device
        dtype = base_layer.weight.dtype
        self.lora_A = nn.Parameter(torch.empty(self.in_features, self.r, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.empty(self.r, self.out_features, device=device, dtype=dtype))
        self.reset_lora_parameters()

        for param in self.base_layer.parameters():
            param.requires_grad = False

    def reset_lora_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_output = self.base_layer(x)
        lora_hidden = torch.matmul(self.lora_dropout(x), self.lora_A)
        lora_update = torch.matmul(lora_hidden, self.lora_B)
        return base_output + self.scaling * lora_update


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen NLG FT/LoRA with confidence variables.")
    parser.add_argument("--task", choices=["e2e", "webnlg", "dart"], required=True)
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--method", choices=["ft", "lora"], default="lora")
    parser.add_argument("--prompt_variant", choices=["strict", "abstain"], default="strict")
    parser.add_argument("--prompt_variants", nargs="+", choices=["strict", "abstain"], default=None)
    parser.add_argument("--max_train_examples", type=int, default=128)
    parser.add_argument("--max_eval_examples", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--max_length", type=int, default=384)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--lora_r", type=int, default=4)
    parser.add_argument("--lora_alpha", type=float, default=32.0)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--calibration_bins", type=int, default=10)
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


def resolve_device(requested_device: Optional[str]) -> torch.device:
    if requested_device:
        if requested_device.startswith("cuda"):
            ensure_cuda_supported()
        return torch.device(requested_device)
    if torch.cuda.is_available() and ensure_cuda_supported(raise_on_unsupported=False):
        return torch.device("cuda")
    return torch.device("cpu")


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def save_jsonl(path: Path, rows: Sequence[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[_|]", " ", text)
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def ordered_unique(items: Sequence[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def parse_e2e_mr(text: str) -> List[Tuple[str, str]]:
    pairs = re.findall(r"([^,\[]+)\[([^\]]+)\]", text)
    return [(slot.strip(), value.strip()) for slot, value in pairs]


def format_e2e_structured_input(mr: str) -> Tuple[str, List[str]]:
    facts = parse_e2e_mr(mr)
    lines = [f"{slot} = {value}" for slot, value in facts]
    fact_values = [value for _, value in facts]
    return "\n".join(lines), fact_values


def camel_or_relation_tokens(text: str) -> List[str]:
    text = text.replace("_", " ")
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    return [token for token in normalize_text(text).split() if token]


def format_triples(triples: Sequence[Sequence[str] | str]) -> Tuple[str, List[Tuple[str, str, str]]]:
    cleaned: List[Tuple[str, str, str]] = []
    lines: List[str] = []
    for triple in triples:
        if isinstance(triple, str):
            parts = [part.strip().strip('"') for part in triple.split("|")]
            if len(parts) != 3:
                continue
            subj, rel, obj = parts
        else:
            if len(triple) != 3:
                continue
            subj, rel, obj = (str(item).strip().strip('"') for item in triple)
        cleaned.append((subj, rel, obj))
        lines.append(f"{subj} | {rel} | {obj}")
    return "\n".join(lines), cleaned


def task_loader_spec(task: str) -> Tuple[str, Optional[str]]:
    if task == "e2e":
        return "GEM/e2e_nlg", "default"
    if task == "webnlg":
        return "GEM/web_nlg", "en"
    if task == "dart":
        return "GEM/dart", "default"
    raise ValueError(task)


def load_parquet_splits(dataset_name: str, dataset_config: Optional[str]):
    api_url = f"https://huggingface.co/api/datasets/{dataset_name}/parquet"
    with urlopen(api_url) as response:
        payload = json.loads(response.read().decode("utf-8"))

    if isinstance(payload, dict) and "parquet_files" in payload:
        parquet_files = payload["parquet_files"]
        grouped: Dict[str, Dict[str, List[str]]] = {}
        for item in parquet_files:
            config_name = item.get("config") or "default"
            split_name = item.get("split")
            url = item.get("url")
            if not split_name or not url:
                continue
            grouped.setdefault(config_name, {}).setdefault(split_name, []).append(url)
    else:
        grouped = payload

    config_name = dataset_config or "default"
    if config_name not in grouped:
        available = sorted(grouped.keys())
        raise ValueError(f"Config {config_name!r} not found for {dataset_name}. Available configs: {available}")

    split_map = grouped[config_name]
    data_files = {}
    for split_name in ("train", "validation"):
        urls = split_map.get(split_name)
        if not urls:
            raise ValueError(f"Split {split_name!r} not found for {dataset_name}/{config_name}. Available splits: {sorted(split_map.keys())}")
        data_files[split_name] = urls

    train_frames = [pd.read_parquet(url) for url in data_files["train"]]
    valid_frames = [pd.read_parquet(url) for url in data_files["validation"]]
    train_df = pd.concat(train_frames, ignore_index=True)
    valid_df = pd.concat(valid_frames, ignore_index=True)
    return HFDataset.from_pandas(train_df, preserve_index=False), HFDataset.from_pandas(valid_df, preserve_index=False)


def build_rows(task: str, split_rows: Sequence[Dict], max_examples: int) -> List[Dict]:
    rows: List[Dict] = []
    for idx, example in enumerate(split_rows):
        if len(rows) >= max_examples:
            break
        if task == "e2e":
            mr = str(example.get("meaning_representation") or example.get("input") or example.get("mr") or "").strip()
            target = str(example.get("target") or "").strip()
            references = [str(item).strip() for item in (example.get("references") or []) if str(item).strip()]
            if target and target not in references:
                references = [target] + references
            if not mr or not target:
                continue
            structured_input, fact_values = format_e2e_structured_input(mr)
            rows.append(
                {
                    "example_id": str(example.get("gem_id", f"{task}-{idx}")),
                    "structured_input": structured_input,
                    "target": target,
                    "references": references or [target],
                    "fact_values": fact_values,
                    "task": task,
                }
            )
        elif task == "webnlg":
            triples = example.get("input") or []
            target = str(example.get("target") or "").strip()
            references = [str(item).strip() for item in (example.get("references") or []) if str(item).strip()]
            if target and target not in references:
                references = [target] + references
            structured_input, cleaned_triples = format_triples(triples)
            if not structured_input or not target:
                continue
            rows.append(
                {
                    "example_id": str(example.get("gem_id", f"{task}-{idx}")),
                    "structured_input": structured_input,
                    "target": target,
                    "references": references or [target],
                    "triples": cleaned_triples,
                    "task": task,
                }
            )
        elif task == "dart":
            triples = example.get("tripleset") or []
            target = str(example.get("target") or "").strip()
            references = [str(item).strip() for item in (example.get("references") or []) if str(item).strip()]
            if target and target not in references:
                references = [target] + references
            structured_input, cleaned_triples = format_triples(triples)
            if not structured_input or not target:
                continue
            rows.append(
                {
                    "example_id": str(example.get("gem_id", f"{task}-{idx}")),
                    "structured_input": structured_input,
                    "target": target,
                    "references": references or [target],
                    "triples": cleaned_triples,
                    "task": task,
                }
            )
    return rows


def format_train_prompt(task: str, structured_input: str) -> str:
    return (
        "You are given structured data. Write one fluent English sentence that verbalizes the input faithfully.\n\n"
        f"Task: {task}\n"
        f"Input:\n{structured_input}\n\n"
        "Respond in exactly this format:\n"
        "Answer: <one sentence>\n"
        "Confidence: <integer 0-100>\n"
    )


def format_eval_prompt(task: str, structured_input: str, prompt_variant: str) -> str:
    abstain_line = (
        ' If you are not confident you can faithfully verbalize it, you may answer "I don\'t know".'
        if prompt_variant == "abstain"
        else ""
    )
    return (
        "You are given structured data. Write one fluent English sentence that verbalizes the input faithfully."
        f"{abstain_line}\n\n"
        f"Task: {task}\n"
        f"Input:\n{structured_input}\n\n"
        "Respond in exactly this format:\n"
        "Answer: <one sentence"
        + (" or I don't know" if prompt_variant == "abstain" else "")
        + ">\n"
        "Confidence: <integer 0-100>\n"
    )


def build_supervised_target(answer: str) -> str:
    return f"Answer: {answer}\nConfidence:"


class SupervisedNLGDataset(TorchDataset):
    def __init__(self, rows: Sequence[Dict], tokenizer, max_length: int):
        self.rows = list(rows)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.rows[idx]
        prompt = format_train_prompt(row["task"], row["structured_input"])
        target = build_supervised_target(row["target"])
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        target_ids = self.tokenizer(target, add_special_tokens=False)["input_ids"]
        eos_id = self.tokenizer.eos_token_id
        input_ids = prompt_ids + target_ids + ([eos_id] if eos_id is not None else [])
        labels = [-100] * len(prompt_ids) + target_ids + ([-100] if eos_id is not None else [])
        input_ids = input_ids[: self.max_length]
        labels = labels[: self.max_length]
        attention_mask = [1] * len(input_ids)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class CausalCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        max_len = max(item["input_ids"].shape[0] for item in batch)
        input_ids, attention_mask, labels = [], [], []
        for item in batch:
            seq_len = item["input_ids"].shape[0]
            pad_len = max_len - seq_len
            input_ids.append(torch.cat([item["input_ids"], torch.full((pad_len,), self.pad_token_id, dtype=torch.long)]))
            attention_mask.append(torch.cat([item["attention_mask"], torch.zeros(pad_len, dtype=torch.long)]))
            labels.append(torch.cat([item["labels"], torch.full((pad_len,), -100, dtype=torch.long)]))
        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_mask),
            "labels": torch.stack(labels),
        }


def infer_attention_targets(model: nn.Module) -> Tuple[str, ...]:
    linear_leaf_names = {
        module_name.rsplit(".", 1)[-1]
        for module_name, module in model.named_modules()
        if isinstance(module, nn.Linear) and module_name
    }
    conv1d_leaf_names = {
        module_name.rsplit(".", 1)[-1]
        for module_name, module in model.named_modules()
        if isinstance(module, Conv1D) and module_name
    }
    if {"q_proj", "v_proj"}.issubset(linear_leaf_names):
        return ("q_proj", "v_proj")
    if {"query", "value"}.issubset(linear_leaf_names):
        return ("query", "value")
    if "c_attn" in conv1d_leaf_names:
        return ("c_attn",)
    raise ValueError(f"Could not infer attention targets. Linear leaf names include: {sorted(linear_leaf_names)[:30]}")


def transformer_layer_index(module_name: str) -> Optional[int]:
    match = re.search(r"\.(?:layers|h|block)\.(\d+)\.", f".{module_name}.")
    return int(match.group(1)) if match else None


def get_parent_module(root: nn.Module, module_name: str) -> Tuple[nn.Module, str]:
    parts = module_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def inject_lora(model: nn.Module, r: int, alpha: float, dropout: float) -> List[str]:
    attention_targets = set(infer_attention_targets(model))
    replaced: List[str] = []
    for module_name, module in list(model.named_modules()):
        if not isinstance(module, (nn.Linear, Conv1D)):
            continue
        if transformer_layer_index(module_name) is None:
            continue
        leaf = module_name.rsplit(".", 1)[-1]
        if leaf not in attention_targets:
            continue
        parent, child_name = get_parent_module(model, module_name)
        if isinstance(module, nn.Linear):
            replacement = MyLoRALinear.from_linear(module, r, alpha, dropout)
        else:
            replacement = MyLoRAConv1D(module, r=r, lora_alpha=alpha, lora_dropout=dropout)
        setattr(parent, child_name, replacement)
        replaced.append(module_name)
    if not replaced:
        raise ValueError("No LoRA modules matched inferred attention targets.")
    return replaced


def mark_trainable(model: nn.Module, method: str) -> None:
    if method == "ft":
        for param in model.parameters():
            param.requires_grad = True
        return
    for name, param in model.named_parameters():
        is_lora = "lora_A" in name or "lora_B" in name
        is_head = "lm_head" in name
        param.requires_grad = is_lora or is_head


def parse_generation(text: str) -> Dict:
    answer_match = re.search(r"Answer:\s*(.+)", text, re.IGNORECASE)
    conf_match = re.search(r"Confidence:\s*([0-9]{1,3})", text, re.IGNORECASE)
    answer = answer_match.group(1).strip() if answer_match else ""
    if "\n" in answer:
        answer = answer.splitlines()[0].strip()
    conf_raw = conf_match.group(1).strip() if conf_match else ""
    conf_int = None
    if conf_raw:
        try:
            conf_int = max(0, min(100, int(conf_raw)))
        except ValueError:
            conf_int = None
    confidence = float(conf_int) / 100.0 if conf_int is not None else 0.0
    abstained = normalize_text(answer) in {
        "i don t know",
        "dont know",
        "do not know",
        "unknown",
        "not sure",
        "i am not sure",
    }
    return {
        "parsed_answer": answer,
        "confidence_raw": conf_raw,
        "confidence_int": conf_int,
        "confidence": confidence,
        "parse_success": bool(answer_match and conf_match and conf_int is not None),
        "abstained": abstained,
    }


def lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    if not a or not b:
        return 0
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1]


def rouge_l_f1(pred: str, ref: str) -> float:
    pred_tokens = normalize_text(pred).split()
    ref_tokens = normalize_text(ref).split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    lcs = lcs_length(pred_tokens, ref_tokens)
    prec = lcs / max(1, len(pred_tokens))
    rec = lcs / max(1, len(ref_tokens))
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def ngrams(tokens: Sequence[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(max(0, len(tokens) - n + 1)))


def corpus_bleu(predictions: Sequence[str], references: Sequence[Sequence[str]], max_order: int = 4) -> float:
    matches = [0] * max_order
    totals = [0] * max_order
    pred_len = 0
    ref_len = 0
    for pred, refs in zip(predictions, references):
        pred_tokens = normalize_text(pred).split()
        ref_tokens_list = [normalize_text(ref).split() for ref in refs if normalize_text(ref)]
        if not pred_tokens or not ref_tokens_list:
            continue
        pred_len += len(pred_tokens)
        ref_lens = [len(ref_tokens) for ref_tokens in ref_tokens_list]
        ref_len += min(ref_lens, key=lambda x: (abs(x - len(pred_tokens)), x))
        for order in range(1, max_order + 1):
            pred_ngrams = ngrams(pred_tokens, order)
            totals[order - 1] += sum(pred_ngrams.values())
            max_ref_counts: Counter = Counter()
            for ref_tokens in ref_tokens_list:
                ref_ngrams = ngrams(ref_tokens, order)
                for gram, count in ref_ngrams.items():
                    if count > max_ref_counts[gram]:
                        max_ref_counts[gram] = count
            matches[order - 1] += sum(min(count, max_ref_counts[gram]) for gram, count in pred_ngrams.items())
    if pred_len == 0:
        return 0.0
    precisions = []
    for match, total in zip(matches, totals):
        if total == 0:
            precisions.append(0.0)
        else:
            precisions.append((match + 1.0) / (total + 1.0))
    if min(precisions) <= 0:
        return 0.0
    geo_mean = math.exp(sum(math.log(p) for p in precisions) / max_order)
    bp = 1.0 if pred_len > ref_len else math.exp(1 - (ref_len / max(1, pred_len)))
    return float(bp * geo_mean)


def expected_calibration_error(confidences: np.ndarray, correct: np.ndarray, n_bins: int) -> Tuple[float, float]:
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
    return float(ece), float(mce)


def compute_repetition_ngram_rate(text: str, n: int = 3) -> float:
    tokens = normalize_text(text).split()
    if len(tokens) < n:
        return 0.0
    grams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    if not grams:
        return 0.0
    return 1.0 - (len(set(grams)) / len(grams))


def fact_metrics(row: Dict, answer: str) -> Dict[str, float]:
    norm_answer = normalize_text(answer)
    if row["task"] == "e2e":
        values = [normalize_text(value) for value in row.get("fact_values", []) if normalize_text(value)]
        mentioned = sum(1 for value in values if value in norm_answer)
        total = len(values)
        recall = float(mentioned / total) if total else 0.0
        return {
            "fact_total_count": total,
            "fact_mentioned_count": mentioned,
            "fact_recall": recall,
            "missing_fact_count": max(0, total - mentioned),
        }
    triples = row.get("triples", [])
    total = len(triples)
    covered = 0
    subj_count = 0
    obj_count = 0
    rel_count = 0
    for subj, rel, obj in triples:
        subj_norm = normalize_text(subj)
        obj_norm = normalize_text(obj)
        rel_tokens = camel_or_relation_tokens(rel)
        subj_hit = bool(subj_norm and subj_norm in norm_answer)
        obj_hit = bool(obj_norm and obj_norm in norm_answer)
        rel_hit = any(token in norm_answer.split() for token in rel_tokens) if rel_tokens else False
        subj_count += int(subj_hit)
        obj_count += int(obj_hit)
        rel_count += int(rel_hit)
        if subj_hit and obj_hit:
            covered += 1
    recall = float(covered / total) if total else 0.0
    return {
        "fact_total_count": total,
        "fact_mentioned_count": covered,
        "fact_recall": recall,
        "missing_fact_count": max(0, total - covered),
        "subject_recall": float(subj_count / total) if total else 0.0,
        "object_recall": float(obj_count / total) if total else 0.0,
        "relation_recall": float(rel_count / total) if total else 0.0,
    }


def compute_logprob_features(batch_input_len: int, generated_ids: torch.Tensor, scores: Sequence[torch.Tensor]) -> Dict[str, float]:
    if generated_ids.numel() == 0 or not scores:
        return {
            "sequence_logprob_sum": 0.0,
            "sequence_logprob_mean": 0.0,
            "answer_logprob_mean": 0.0,
            "answer_logprob_min": 0.0,
            "first_token_logprob": 0.0,
            "top1_top2_margin_mean": 0.0,
            "entropy_mean": 0.0,
        }
    token_logprobs: List[float] = []
    margins: List[float] = []
    entropies: List[float] = []
    for token_id, score_t in zip(generated_ids.tolist(), scores):
        log_probs = torch.log_softmax(score_t[0].float(), dim=-1)
        probs = torch.softmax(score_t[0].float(), dim=-1)
        token_logprobs.append(float(log_probs[token_id].item()))
        top2 = torch.topk(log_probs, k=2).values
        margins.append(float((top2[0] - top2[1]).item()) if top2.numel() == 2 else 0.0)
        entropies.append(float((-(probs * log_probs).sum()).item()))
    return {
        "sequence_logprob_sum": float(sum(token_logprobs)),
        "sequence_logprob_mean": float(sum(token_logprobs) / len(token_logprobs)),
        "answer_logprob_mean": float(sum(token_logprobs) / len(token_logprobs)),
        "answer_logprob_min": float(min(token_logprobs)),
        "first_token_logprob": float(token_logprobs[0]),
        "top1_top2_margin_mean": float(sum(margins) / len(margins)),
        "entropy_mean": float(sum(entropies) / len(entropies)),
    }


@torch.no_grad()
def run_eval(
    model,
    tokenizer,
    rows: Sequence[Dict],
    device: torch.device,
    *,
    prompt_variant: str,
    max_new_tokens: int,
    abstention_threshold: float,
    calibration_bins: int,
) -> Tuple[Dict, List[Dict]]:
    model.eval()
    predictions: List[Dict] = []
    exact_correct: List[bool] = []
    confidences: List[float] = []
    parse_successes: List[bool] = []
    rouge_scores: List[float] = []
    fact_recalls: List[float] = []
    seq_logprob_means: List[float] = []
    answer_logprob_means: List[float] = []

    for row in tqdm(rows, desc="eval", leave=False):
        prompt = format_eval_prompt(row["task"], row["structured_input"], prompt_variant)
        batch = tokenizer(prompt, return_tensors="pt").to(device)
        output = model.generate(
            **batch,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )
        full_sequence = output.sequences[0]
        generated_ids = full_sequence[batch["input_ids"].shape[1] :]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        parsed = parse_generation(generated_text)
        parsed_answer = parsed["parsed_answer"]
        references = row["references"]
        norm_answer = normalize_text(parsed_answer)
        norm_refs = [normalize_text(ref) for ref in references]
        exact_match = bool(norm_answer and norm_answer in norm_refs and not parsed["abstained"])
        rouge_best = max((rouge_l_f1(parsed_answer, ref) for ref in references), default=0.0)
        fact_payload = fact_metrics(row, parsed_answer)
        logprob_payload = compute_logprob_features(batch["input_ids"].shape[1], generated_ids.cpu(), output.scores)
        confidence = parsed["confidence"]

        answer_token_len = len(tokenizer(parsed_answer, add_special_tokens=False)["input_ids"]) if parsed_answer else 0
        response_token_len = len(tokenizer(generated_text, add_special_tokens=False)["input_ids"]) if generated_text else 0
        response_tokens = normalize_text(generated_text).split()
        unique_ratio = float(len(set(response_tokens)) / len(response_tokens)) if response_tokens else 0.0
        contains_refusal = "i don't know" in generated_text.lower() or "i do not know" in generated_text.lower()

        prediction = {
            "example_id": row["example_id"],
            "task": row["task"],
            "prompt_variant": prompt_variant,
            "structured_input": row["structured_input"],
            "target": row["target"],
            "references": references,
            "raw_response": generated_text,
            "parsed_answer": parsed_answer,
            "normalized_answer": norm_answer,
            "exact_match": exact_match,
            "is_correct": exact_match,
            "reference_rouge_l": rouge_best,
            "confidence_raw_text": parsed["confidence_raw"],
            "confidence_int": parsed["confidence_int"],
            "confidence": confidence,
            "parse_success": parsed["parse_success"],
            "abstained_by_text": parsed["abstained"],
            "contains_refusal_phrase": contains_refusal,
            "wrong_but_high_confidence": (not exact_match) and confidence >= 0.80,
            "correct_but_low_confidence": exact_match and confidence <= 0.40,
            "confidence_bin": int(min(4, confidence * 5)),
            "answer_length_chars": len(parsed_answer),
            "answer_length_tokens": answer_token_len,
            "response_length_tokens": response_token_len,
            "num_sentences": max(1, len([p for p in re.split(r"[.!?]+", parsed_answer) if p.strip()])) if parsed_answer else 0,
            "format_exact_match": bool(re.search(r"^\s*Answer:\s*.+\nConfidence:\s*[0-9]{1,3}\s*$", generated_text, re.IGNORECASE)),
            "repetition_ngram_rate": compute_repetition_ngram_rate(parsed_answer),
            "unique_token_ratio": unique_ratio,
            **fact_payload,
            **logprob_payload,
        }
        predictions.append(prediction)
        exact_correct.append(exact_match)
        confidences.append(confidence)
        parse_successes.append(parsed["parse_success"])
        rouge_scores.append(rouge_best)
        fact_recalls.append(fact_payload["fact_recall"])
        seq_logprob_means.append(logprob_payload["sequence_logprob_mean"])
        answer_logprob_means.append(logprob_payload["answer_logprob_mean"])

    conf = np.asarray(confidences, dtype=np.float64)
    correct = np.asarray(exact_correct, dtype=bool)
    attempted = conf >= float(abstention_threshold)
    ece, mce = expected_calibration_error(conf, correct, calibration_bins)

    metrics = {
        "primary_metric": "bleu",
        "primary_metric_value": corpus_bleu(
            [pred["parsed_answer"] for pred in predictions],
            [pred["references"] for pred in predictions],
        ),
        "accuracy": float(np.mean(correct)) if len(correct) else 0.0,
        "exact_match_rate": float(np.mean(correct)) if len(correct) else 0.0,
        "bleu": corpus_bleu(
            [pred["parsed_answer"] for pred in predictions],
            [pred["references"] for pred in predictions],
        ),
        "rouge_l": float(np.mean(rouge_scores)) if rouge_scores else 0.0,
        "parse_rate": float(np.mean(parse_successes)) if parse_successes else 0.0,
        "fact_recall_mean": float(np.mean(fact_recalls)) if fact_recalls else 0.0,
        "mean_confidence": float(np.mean(conf)) if len(conf) else 0.0,
        "median_confidence": float(median(confidences)) if confidences else 0.0,
        "confidence_correct_mean": float(np.mean(conf[correct])) if np.any(correct) else 0.0,
        "confidence_wrong_mean": float(np.mean(conf[~correct])) if np.any(~correct) else 0.0,
        "ece": ece,
        "mce": mce,
        "calibration_accuracy": float(1.0 - ece),
        "high_conf_wrong_rate": float(np.mean([pred["wrong_but_high_confidence"] for pred in predictions])) if predictions else 0.0,
        "low_conf_correct_rate": float(np.mean([pred["correct_but_low_confidence"] for pred in predictions])) if predictions else 0.0,
        "abstention_threshold": float(abstention_threshold),
        "abstention_rate": float(np.mean(~attempted)) if len(attempted) else 0.0,
        "coverage": float(np.mean(attempted)) if len(attempted) else 0.0,
        "selective_accuracy": float(np.mean(correct[attempted])) if np.any(attempted) else 0.0,
        "selective_rouge_l": float(np.mean(np.asarray(rouge_scores)[attempted])) if np.any(attempted) else 0.0,
        "abstained_by_text_rate": float(np.mean([pred["abstained_by_text"] for pred in predictions])) if predictions else 0.0,
        "mean_answer_tokens": float(np.mean([pred["answer_length_tokens"] for pred in predictions])) if predictions else 0.0,
        "mean_response_tokens": float(np.mean([pred["response_length_tokens"] for pred in predictions])) if predictions else 0.0,
        "mean_sequence_logprob": float(np.mean(seq_logprob_means)) if seq_logprob_means else 0.0,
        "mean_answer_logprob": float(np.mean(answer_logprob_means)) if answer_logprob_means else 0.0,
    }
    return metrics, predictions


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_variants = ordered_unique(args.prompt_variants or [args.prompt_variant])
    multi_variant = len(prompt_variants) > 1
    device = resolve_device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
        if getattr(model, "generation_config", None) is not None:
            model.generation_config.pad_token_id = tokenizer.pad_token_id
    if args.method == "lora":
        replaced = inject_lora(model, args.lora_r, args.lora_alpha, args.lora_dropout)
    else:
        replaced = []
    mark_trainable(model, args.method)
    model.to(device)

    dataset_name, dataset_config = task_loader_spec(args.task)
    train_split, eval_split = load_parquet_splits(dataset_name, dataset_config)
    train_split = train_split.shuffle(seed=args.seed)
    eval_split = eval_split.shuffle(seed=args.seed)
    train_rows = build_rows(args.task, [train_split[idx] for idx in range(len(train_split))], args.max_train_examples)
    eval_rows = build_rows(args.task, [eval_split[idx] for idx in range(len(eval_split))], args.max_eval_examples)

    train_dataset = SupervisedNLGDataset(train_rows, tokenizer, args.max_length)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=CausalCollator(tokenizer.pad_token_id),
    )

    learning_rate = args.learning_rate
    if learning_rate is None:
        learning_rate = 5e-5 if args.method == "ft" else 2e-4
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = AdamW(trainable_params, lr=learning_rate, weight_decay=args.weight_decay)
    total_steps = max(1, (len(train_loader) * args.epochs) // max(1, args.gradient_accumulation_steps))
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    base_train_config = {
        "task_name": f"nlg_{args.task}",
        "dataset_task": args.task,
        "dataset_name": dataset_name,
        "dataset_config": dataset_config,
        "model_name": args.model_name,
        "method": args.method,
        "prompt_variants": prompt_variants,
        "max_train_examples": args.max_train_examples,
        "max_eval_examples": args.max_eval_examples,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": learning_rate,
        "max_length": args.max_length,
        "max_new_tokens": args.max_new_tokens,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "calibration_bins": args.calibration_bins,
        "abstention_threshold": args.abstention_threshold,
        "seed": args.seed,
        "train_examples": len(train_rows),
        "eval_examples": len(eval_rows),
        "replaced_modules": replaced,
    }
    param_counts = count_trainable_parameters(model)
    if not multi_variant:
        single_config = dict(base_train_config)
        single_config["prompt_variant"] = prompt_variants[0]
        write_json(output_dir / "train_config.json", single_config)
        write_json(output_dir / "parameter_count.json", param_counts)
    else:
        write_json(output_dir / "train_config.json", base_train_config)
        write_json(output_dir / "parameter_count.json", param_counts)

    model.train()
    for epoch in range(args.epochs):
        optimizer.zero_grad(set_to_none=True)
        progress = tqdm(train_loader, desc=f"train epoch {epoch+1}/{args.epochs}", leave=False)
        for step, batch in enumerate(progress):
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / max(1, args.gradient_accumulation_steps)
            loss.backward()
            if (step + 1) % max(1, args.gradient_accumulation_steps) == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                progress.set_postfix(loss=float(loss.detach().cpu()) * max(1, args.gradient_accumulation_steps))

    for prompt_variant in prompt_variants:
        variant_output_dir = (
            output_dir / f"qwen_{args.task}_{prompt_variant}_{args.method}"
            if multi_variant
            else output_dir
        )
        variant_output_dir.mkdir(parents=True, exist_ok=True)
        variant_train_config = dict(base_train_config)
        variant_train_config["prompt_variant"] = prompt_variant
        write_json(variant_output_dir / "train_config.json", variant_train_config)
        write_json(variant_output_dir / "parameter_count.json", param_counts)

        eval_metrics, predictions = run_eval(
            model,
            tokenizer,
            eval_rows,
            device,
            prompt_variant=prompt_variant,
            max_new_tokens=args.max_new_tokens,
            abstention_threshold=args.abstention_threshold,
            calibration_bins=args.calibration_bins,
        )

        write_json(variant_output_dir / "metrics.json", {"final_id": eval_metrics})
        save_jsonl(variant_output_dir / "id_predictions.jsonl", predictions)
        print(f"Done: {variant_output_dir}")
        print(
            "NLG "
            f"bleu={eval_metrics['bleu']:.4f} "
            f"rouge_l={eval_metrics['rouge_l']:.4f} "
            f"parse_rate={eval_metrics['parse_rate']:.4f} "
            f"mean_conf={eval_metrics['mean_confidence']:.4f}"
        )


if __name__ == "__main__":
    main()
