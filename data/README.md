# Data

This directory contains lightweight dataset previews and metadata. Full datasets are loaded through Hugging Face `datasets` by the scripts in `code/`; large raw datasets are not committed.

## Included Preview Files

- `glue_sst2/`: SST-2 preview splits and metadata.
- `glue_rte/`: RTE preview splits and metadata.
- `glue_mrpc/`: MRPC preview splits and metadata.
- `glue_cola/`: CoLA preview splits and metadata.

Each GLUE preview folder contains `train_preview.jsonl`, `validation_preview.jsonl`, `test_preview.jsonl`, and `metadata.json`.

## Datasets Used In The Report

- NLU: GLUE SST-2, MRPC, RTE, and CoLA.
- Vision: CIFAR-10, Beans, and `timm/oxford-iiit-pet`.
- Audio: Minds14-EN, Speech Commands, and SUPERB Emotion Recognition.
- Reliability/calibration: GLUE tasks with confidence metrics derived from model probabilities.
- NLG draft results: E2E and WebNLG, with DART used only for smoke tests.

## Recreating Data Locally

GLUE previews can be regenerated with:

```bash
python code/reimpl/prepare_glue.py --task_name sst2 --output_dir data/glue_sst2
python code/reimpl/prepare_glue.py --task_name rte --output_dir data/glue_rte
python code/reimpl/prepare_glue.py --task_name mrpc --output_dir data/glue_mrpc
python code/reimpl/prepare_glue.py --task_name cola --output_dir data/glue_cola
```

The vision and audio scripts download their datasets from Hugging Face when first run, subject to each dataset's license and access requirements.
