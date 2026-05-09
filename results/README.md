# Results

This directory contains reproduced metrics, logs, predictions, figures, and compact summary tables for the LoRA vs full fine-tuning project. Heavy model checkpoints were intentionally excluded from the GitHub-ready copy.

## Layout

- `summary_tables/`: Report-level CSV tables matching the draft report's appendix.
- `nlu/`: GLUE task outputs for SST-2, RTE, MRPC, and CoLA, including metrics, logs, predictions, per-task summary tables, and figures.
- `vision/`: ViT results for CIFAR-10, Beans, and Oxford-IIIT Pet.
- `audio/`: wav2vec2 results for Minds14-EN, Speech Commands, and SUPERB ER.
- `reliability/`: ECE, Brier, NLL, and selective-accuracy experiments for GLUE tasks, including rank/placement sweeps.
- `logs/`: Scheduler stdout/stderr logs copied from the original experiment workspace.

## Report Table Mapping

- Draft Table 1, NLU reproduction: `summary_tables/nlu_reproduction.csv`, plus detailed runs under `nlu/by_task/`.
- Draft Table 2, multimodal extension: `summary_tables/multimodal_extension.csv`, with detailed vision/audio results under `vision/` and `audio/`.
- Draft Table 3, reliability: `summary_tables/reliability_glue.csv`, with detailed calibration outputs under `reliability/exp1_ft_vs_lora/`.
- Draft Table 4, NLG: `summary_tables/nlg_full.csv`. No E2E/WebNLG source scripts were found in `/home/tx88/4782finalproject`, so this table records the draft-report values for completeness.

## Notes

Checkpoint directories and weight files such as `trainable_state.pt` are excluded to keep the repository lightweight. Re-run the commands in `code/` to regenerate checkpoints locally.
