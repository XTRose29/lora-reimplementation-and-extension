# My Vision LoRA Reimplementation

This folder extends our CS 4782 LoRA reimplementation into vision.

We reuse the custom LoRA layer from `../reimpl/my_lora.py` and compare:

- `LoRA`: freeze the ViT backbone and train low-rank updates plus the classifier head.
- `FT`: full fine-tuning of the entire model.

## Selected Vision Tasks

We choose three practical image understanding tasks that are easy to rerun with Hugging Face datasets:

- `cifar10`: natural image classification, 10 classes.
- `beans`: plant disease classification, 3 classes.
- `oxford_iiit_pet`: fine-grained pet breed classification, 37 classes, loaded from Hugging Face dataset `timm/oxford-iiit-pet`.

These give us a small but diverse comparison set across natural images, agriculture, and fine-grained recognition.

## Setup

Run from the repository root. The cluster scripts in this repo default to the existing `deepseek_env` conda environment.

```bash
python3 my_Vision/train_my_lora_vision.py --help
```

## Quick Smoke Run

```bash
python3 my_Vision/train_my_lora_vision.py \
  --task_name beans \
  --method lora \
  --output_dir my_Vision/results/beans_lora_smoke \
  --epochs 1 \
  --batch_size 8 \
  --learning_rate 5e-4 \
  --max_train_samples 64 \
  --max_eval_samples 64
```

Evaluate the saved checkpoint:

```bash
python3 my_Vision/evaluate_my_lora_vision.py \
  --checkpoint_dir my_Vision/results/beans_lora_smoke/checkpoint \
  --output_dir my_Vision/results/beans_lora_smoke \
  --max_eval_samples 64
```

## Run One Task

```bash
./my_Vision/run_vision_task.sh --task-name cifar10
./my_Vision/run_vision_task.sh --task-name beans
./my_Vision/run_vision_task.sh --task-name oxford_iiit_pet
```

Quick version:

```bash
./my_Vision/run_vision_task.sh --task-name beans --quick
```

## Run All Three Tasks

```bash
./my_Vision/run_vision_task.sh --all
```

Quick version:

```bash
./my_Vision/run_vision_task.sh --all --quick
```

## Outputs

For each experiment we write:

- `metrics.json`
- `eval_predictions.json`
- `parameter_count.json`
- `train_config.json`
- `train_log.jsonl`
- `checkpoint/trainable_state.pt`
- `checkpoint/lora_config.json`

Aggregated summaries are written to:

- `my_Vision/results/by_task/<task>/summary_table.csv`
- `my_Vision/results/three_task_table/summary_table.csv`
