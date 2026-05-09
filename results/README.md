# Results

This directory contains reproduced metrics, logs, predictions, figures, and compact summary tables for the LoRA vs full fine-tuning project. Heavy model checkpoints were intentionally excluded from the GitHub-ready copy.

## Layout

- `summary_tables/`: Report-level CSV tables matching the final project report.
- `nlu/`: GLUE task outputs for SST-2, RTE, MRPC, and CoLA, including metrics, logs, predictions, per-task summary tables, and figures.
- `vision/`: ViT results for CIFAR-10, Beans, and Oxford-IIIT Pet.
- `audio/`: wav2vec2 results for Minds14-EN, Speech Commands, and SUPERB ER.
- `reliability/`: ECE, Brier, NLL, and selective-accuracy experiments for GLUE tasks, including rank/placement sweeps.
- `nlg/`: Qwen NLG full and smoke summary tables plus downloaded run artifacts.
- `logs/`: Scheduler stdout/stderr logs copied from the original experiment workspace.

## Report Table Mapping

- Report Table 1, NLU reproduction: `summary_tables/nlu_reproduction.csv`, plus detailed runs under `nlu/by_task/`.
- Report Table 2, multimodal extension: `summary_tables/multimodal_extension.csv`, with detailed vision/audio results under `vision/` and `audio/`.
- Report Table 3, reliability: `summary_tables/reliability_glue.csv`, with detailed calibration outputs under `reliability/exp1_ft_vs_lora/`.
- Report Table 4, NLG: `summary_tables/nlg_full.csv`, with task-level summaries under `nlg/full/e2e/` and `nlg/full/webnlg/`.

## Notes

Checkpoint directories and weight files such as `trainable_state.pt` are excluded to keep the repository lightweight. Re-run the commands in `code/` to regenerate checkpoints locally.
