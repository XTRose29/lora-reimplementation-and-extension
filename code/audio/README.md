# My Audio LoRA Reimplementation

This folder extends our CS 4782 LoRA reimplementation into audio classification.

We reuse the custom LoRA layer from `../reimpl/my_lora.py` and compare:

- `LoRA`: freeze the backbone and train low-rank updates plus the classification head.
- `FT`: full fine-tuning of the entire model.

## Selected Audio Tasks

- `speech_commands`: keyword spotting from `google/speech_commands` with config `v0.02`
- `minds14_en`: spoken intent classification from `PolyAI/minds14` with config `en-US`
- `superb_er`: speech emotion recognition from `s3prl/superb` with config `er`

## Model

By default we use:

```text
facebook/wav2vec2-base
```

The LoRA injection targets the audio attention projections:

```text
q_proj, v_proj
```

## Setup

Run from the repository root. The Slurm script defaults to the existing `deepseek_env` conda environment.

## Quick Smoke Run

```bash
python my_Audio/train_my_lora_audio.py \
  --task_name speech_commands \
  --method lora \
  --output_dir my_Audio/results/speech_commands_lora_smoke \
  --epochs 1 \
  --batch_size 4 \
  --learning_rate 1e-4 \
  --max_train_samples 64 \
  --max_eval_samples 64
```

## Run One Task

```bash
./my_Audio/run_audio_task.sh --task-name speech_commands
./my_Audio/run_audio_task.sh --task-name minds14_en
./my_Audio/run_audio_task.sh --task-name superb_er
```

Quick version:

```bash
./my_Audio/run_audio_task.sh --task-name minds14_en --quick
```

## Run All Three Tasks

```bash
./my_Audio/run_audio_task.sh --all
```

## Output Columns

The summary CSV is written with this exact order:

```text
experiment,task_name,method,model_name,accuracy,eval_loss,trainable_parameters,# Trainable Parameters,trainable_ratio
```

