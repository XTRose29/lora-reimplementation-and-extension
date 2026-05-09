# Code

This directory contains the re-implementation code used for the CS 4782 LoRA project.

## Layout

- `reimpl/`: Core NLU LoRA/adapter/full fine-tuning implementation used by the other scripts.
- `nlu/`: NLU experiment scripts and launch helpers for GLUE-style tasks.
- `vision/`: ViT full fine-tuning and LoRA experiments for CIFAR-10, Beans, and Oxford-IIIT Pet.
- `audio/`: wav2vec2 full fine-tuning and LoRA experiments for Minds14-EN, Speech Commands, and SUPERB ER.
- `reliability/`: Calibration, ECE, Brier, NLL, and selective-accuracy experiments on GLUE tasks.
- `task_probed_lora/`: Extension code for task-probed LoRA placement experiments.
- `newimpl/`: Compatibility package for task-probed LoRA imports.
- `loralib/`: Local copy of the reference LoRA library files used for comparison/context.

## Environment

Install dependencies from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r code/requirements.txt
```

The scripts expect the repository `code/` directory on `PYTHONPATH` when run directly:

```bash
export PYTHONPATH="$PWD/code:$PYTHONPATH"
```

## Example Commands

```bash
python code/reimpl/train_my_lora_nlu.py --task_name sst2 --method lora --output_dir results/nlu/example_sst2_lora
python code/vision/train_my_lora_vision.py --task_name cifar10 --method lora --output_dir results/vision/example_cifar10_lora
python code/audio/train_my_lora_audio.py --task_name speech_commands --method lora --output_dir results/audio/example_speech_commands_lora
python code/reliability/run_cola_reliability.py --task_name cola --method lora --output_dir results/reliability/example_cola_lora
```

Use the shell scripts in each subdirectory for the larger experiment batches reported in the paper draft.
