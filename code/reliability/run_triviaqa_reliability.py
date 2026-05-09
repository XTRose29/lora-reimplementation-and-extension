"""Train/evaluate FT and LoRA reliability on a small TriviaQA answer-ranking task.

TriviaQA is open-ended QA, while the GLUE reliability script is classification.
This script turns validation questions into candidate-answer ranking examples:
one gold candidate plus sampled negatives from other validation answers. The
model predicts whether a candidate answers the question, and evaluation scores
the top candidate per question.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "reimpl") not in sys.path:
    sys.path.insert(0, str(ROOT / "reimpl"))

import numpy as np
import torch
from datasets import Dataset, load_dataset
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
from transformers import get_linear_schedule_with_warmup

from reimpl.my_lora import MyLoRALinear, count_trainable_parameters, save_trainable_state

try:
    from .metrics import softmax
except ImportError:
    from metrics import softmax


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TriviaQA answer-ranking calibration/abstention experiment.")
    parser.add_argument("--model_name", default="roberta-base")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--method", choices=["ft", "lora"], default="lora")
    parser.add_argument("--dataset_name", default="trivia_qa")
    parser.add_argument("--dataset_config", default="rc.nocontext")
    parser.add_argument("--source_split", default="validation")
    parser.add_argument("--max_train_questions", type=int, default=1000)
    parser.add_argument("--max_eval_questions", type=int, default=1000)
    parser.add_argument("--num_negative_candidates", type=int, default=3)
    parser.add_argument("--lora_r", type=int, default=4)
    parser.add_argument("--lora_alpha", type=float, default=32.0)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--lora_placement", choices=["attention", "mlp", "attention_mlp"], default="attention")
    parser.add_argument("--attention_targets", default="auto")
    parser.add_argument("--layer_indices", default="all")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--max_length", type=int, default=192)
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
    raise ValueError(f"Could not infer attention targets. Linear leaf names include: {sorted(linear_leaf_names)[:30]}")


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
    attention_markers = (".attention.self.", ".self_attn.", ".attention.", ".attn.")
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
    for module_name, module in list(model.named_modules()):
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


def answer_text(example: Dict) -> str:
    answer = example.get("answer", {})
    if isinstance(answer, dict):
        aliases = answer.get("aliases") or answer.get("normalized_aliases") or []
        if aliases:
            return str(aliases[0])
        if answer.get("value"):
            return str(answer["value"])
        if answer.get("normalized_value"):
            return str(answer["normalized_value"])
    return str(answer)


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return " ".join(text.split())


def build_candidate_dataset(raw_split, *, max_train_questions: int, max_eval_questions: int, num_negatives: int, seed: int):
    needed = max_train_questions + max_eval_questions
    if needed > len(raw_split):
        needed = len(raw_split)
    selected = raw_split.select(range(needed))
    questions = [str(row["question"]) for row in selected]
    answers = [answer_text(row) for row in selected]
    all_answers = [answer for answer in answers if answer.strip()]
    rng = random.Random(seed)

    def make_rows(start: int, end: int) -> List[Dict]:
        rows: List[Dict] = []
        for local_group_id, idx in enumerate(range(start, end)):
            question = questions[idx]
            gold = answers[idx]
            if not question.strip() or not gold.strip():
                continue
            gold_norm = normalize_answer(gold)
            negatives = [candidate for candidate in all_answers if normalize_answer(candidate) != gold_norm]
            sampled = rng.sample(negatives, k=min(num_negatives, len(negatives)))
            candidates = [(gold, 1)] + [(negative, 0) for negative in sampled]
            rng.shuffle(candidates)
            for candidate_idx, (candidate, label) in enumerate(candidates):
                rows.append(
                    {
                        "group_id": local_group_id,
                        "candidate_idx": candidate_idx,
                        "question": question,
                        "candidate_answer": candidate,
                        "text": f"Question: {question}\nCandidate answer: {candidate}",
                        "labels": int(label),
                    }
                )
        return rows

    train_rows = make_rows(0, min(max_train_questions, needed))
    eval_rows = make_rows(min(max_train_questions, needed), needed)
    return train_rows, eval_rows


def tokenize_dataset(rows: List[Dict], tokenizer, max_length: int) -> Dataset:
    dataset = Dataset.from_list(rows)

    def tokenize_batch(batch):
        tokenized = tokenizer(batch["text"], truncation=True, max_length=max_length)
        tokenized["labels"] = batch["labels"]
        tokenized["group_id"] = batch["group_id"]
        tokenized["candidate_idx"] = batch["candidate_idx"]
        return tokenized

    return dataset.map(tokenize_batch, batched=True, remove_columns=dataset.column_names)


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


@torch.no_grad()
def run_eval(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    *,
    calibration_bins: int,
    abstention_threshold: float,
) -> Tuple[Dict[str, float], List[Dict]]:
    model.eval()
    candidate_records: List[Dict] = []
    losses: List[float] = []
    for batch in dataloader:
        group_ids = batch.pop("group_id").detach().cpu().numpy().astype(int).tolist()
        candidate_indices = batch.pop("candidate_idx").detach().cpu().numpy().astype(int).tolist()
        labels = batch["labels"].detach().cpu().numpy().astype(int).tolist()
        batch = {key: value.to(device) for key, value in batch.items()}
        outputs = model(**batch)
        if getattr(outputs, "loss", None) is not None:
            losses.append(float(outputs.loss.detach().cpu()))
        logits = outputs.logits.detach().cpu().numpy()
        probs = softmax(logits)
        correct_probs = probs[:, 1]
        for group_id, candidate_idx, label, logit, prob in zip(group_ids, candidate_indices, labels, logits, correct_probs):
            candidate_records.append(
                {
                    "group_id": group_id,
                    "candidate_idx": candidate_idx,
                    "label": int(label),
                    "logits": logit.tolist(),
                    "correct_prob": float(prob),
                }
            )

    groups: Dict[int, List[Dict]] = {}
    for record in candidate_records:
        groups.setdefault(record["group_id"], []).append(record)

    top_correct: List[bool] = []
    top_confidences: List[float] = []
    reciprocal_ranks: List[float] = []
    prediction_dump: List[Dict] = []

    for group_id, records in sorted(groups.items()):
        scores = np.asarray([record["correct_prob"] for record in records], dtype=np.float64)
        group_probs = softmax(scores)
        order = np.argsort(-group_probs)
        top_idx = int(order[0])
        is_correct = records[top_idx]["label"] == 1
        top_correct.append(is_correct)
        top_confidences.append(float(group_probs[top_idx]))
        gold_ranks = [rank + 1 for rank, idx in enumerate(order.tolist()) if records[int(idx)]["label"] == 1]
        reciprocal_ranks.append(1.0 / gold_ranks[0] if gold_ranks else 0.0)
        prediction_dump.append(
            {
                "group_id": group_id,
                "top_candidate_idx": records[top_idx]["candidate_idx"],
                "top_label": records[top_idx]["label"],
                "top_confidence": float(group_probs[top_idx]),
                "candidate_probs": group_probs.tolist(),
            }
        )

    confidences = np.asarray(top_confidences, dtype=np.float64)
    correct = np.asarray(top_correct, dtype=bool)
    attempted = confidences >= float(abstention_threshold)
    ece, mce = expected_calibration_error(confidences, correct, calibration_bins)
    metrics: Dict[str, float] = {
        "primary_metric": "answer_selection_accuracy",
        "primary_metric_value": float(np.mean(correct)) if len(correct) else 0.0,
        "answer_selection_accuracy": float(np.mean(correct)) if len(correct) else 0.0,
        "mrr": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0,
        "ece": ece,
        "mce": mce,
        "calibration_bins": int(calibration_bins),
        "calibration_accuracy": float(1.0 - ece),
        "mean_confidence": float(np.mean(confidences)) if len(confidences) else 0.0,
        "abstention_threshold": float(abstention_threshold),
        "abstention_rate": float(np.mean(~attempted)) if len(attempted) else 0.0,
        "coverage": float(np.mean(attempted)) if len(attempted) else 0.0,
        "selective_accuracy": float(np.mean(correct[attempted])) if np.any(attempted) else 0.0,
        "eval_loss": float(np.mean(losses)) if losses else 0.0,
    }
    model.train()
    return metrics, prediction_dump


def save_jsonl(path: Path, rows: Sequence[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / "checkpoint"
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)
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

    raw = load_dataset(args.dataset_name, args.dataset_config, split=args.source_split)
    train_rows, eval_rows = build_candidate_dataset(
        raw,
        max_train_questions=args.max_train_questions,
        max_eval_questions=args.max_eval_questions,
        num_negatives=args.num_negative_candidates,
        seed=args.seed,
    )
    if not train_rows or not eval_rows:
        raise ValueError("TriviaQA candidate data is empty; increase sample sizes or check dataset fields.")

    encoded_train = tokenize_dataset(train_rows, tokenizer, args.max_length)
    encoded_eval = tokenize_dataset(eval_rows, tokenizer, args.max_length)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_loader = DataLoader(encoded_train, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    eval_loader = DataLoader(encoded_eval, batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    train_config = vars(args).copy()
    train_config.update(
        {
            "task_name": "triviaqa",
            "task_type": "answer_ranking",
            "primary_metric": "answer_selection_accuracy",
            "device": str(device),
            "resolved_learning_rate": lr,
            "attention_targets": list(attention_targets),
            "selected_layers": "all" if selected_layers is None else sorted(selected_layers),
            "replaced_modules": replaced_modules,
            "train_candidate_examples": len(train_rows),
            "eval_candidate_examples": len(eval_rows),
            "parameter_count": count_trainable_parameters(model),
        }
    )
    write_json(output_dir / "train_config.json", train_config)
    write_json(output_dir / "parameter_count.json", train_config["parameter_count"])

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
            batch.pop("group_id")
            batch.pop("candidate_idx")
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

        eval_metrics, _ = run_eval(
            model,
            eval_loader,
            device,
            calibration_bins=args.calibration_bins,
            abstention_threshold=args.abstention_threshold,
        )
        record = {"epoch": epoch, "global_step": global_step, "train_loss": float(np.mean(losses)), **eval_metrics}
        append_jsonl(output_dir / "train_log.jsonl", record)
        current_metric = float(eval_metrics["answer_selection_accuracy"])
        if current_metric > best_metric:
            best_metric = current_metric
            best_metrics = dict(eval_metrics)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            save_trainable_state(model, checkpoint_dir / "trainable_state.pt")
            tokenizer.save_pretrained(checkpoint_dir)
            write_json(checkpoint_dir / "experiment_config.json", train_config)

    id_metrics, predictions = run_eval(
        model,
        eval_loader,
        device,
        calibration_bins=args.calibration_bins,
        abstention_threshold=args.abstention_threshold,
    )
    id_metrics.update({"split": args.source_split, "distribution": "id", "best_primary_metric": best_metric})
    write_json(output_dir / "id_metrics.json", id_metrics)
    save_jsonl(output_dir / "id_predictions.jsonl", predictions)
    final_metrics = {
        "method": args.method,
        "task_name": "triviaqa",
        "model_name": args.model_name,
        **best_metrics,
        "final_id": id_metrics,
    }
    write_json(output_dir / "metrics.json", final_metrics)
    print(f"Done: {output_dir}")
    print(
        f"TriviaQA answer_selection_accuracy={id_metrics['answer_selection_accuracy']:.4f} "
        f"ECE={id_metrics['ece']:.4f} "
        f"abstain={id_metrics['abstention_rate']:.4f}"
    )


if __name__ == "__main__":
    main()
