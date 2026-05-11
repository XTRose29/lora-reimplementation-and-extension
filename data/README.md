# Data

This directory contains lightweight dataset previews and metadata. Full datasets are loaded through Hugging Face `datasets` by the scripts in `code/`; large raw datasets are not committed.

## Included Preview Files

- `glue_sst2/`: SST-2 preview splits and metadata.
- `glue_rte/`: RTE preview splits and metadata.
- `glue_mrpc/`: MRPC preview splits and metadata.
- `glue_cola/`: CoLA preview splits and metadata.

Each GLUE preview folder contains `train_preview.jsonl`, `validation_preview.jsonl`, `test_preview.jsonl`, and `metadata.json`.

## Datasets Used In The Final Report

- NLU and reliability: GLUE SST-2, MRPC, RTE, and CoLA.
- Vision: CIFAR-10, Beans, and Oxford-IIIT Pet.
- Audio: Minds14-EN, Speech Commands, and SUPERB Emotion Recognition.
- NLG: E2E and WebNLG.

## How To Obtain The Datasets

The project does not commit full raw datasets. All full datasets are obtained at run time from Hugging Face using the `datasets` package, and cached in the normal Hugging Face cache directory on the machine running the scripts. Install the project dependencies from the repository root first:

```bash
pip install -r code/requirements.txt
```

The exact dataset identifiers used by the final-report code are:

| Report area | Dataset in report | Hugging Face source used by code | Config |
| --- | --- | --- | --- |
| NLU / reliability | SST-2 | `glue` | `sst2` |
| NLU / reliability | MRPC | `glue` | `mrpc` |
| NLU / reliability | RTE | `glue` | `rte` |
| NLU / reliability | CoLA | `glue` | `cola` |
| NLG | E2E | `GEM/e2e_nlg` | `default` |
| NLG | WebNLG | `GEM/web_nlg` | `en` |
| Vision | CIFAR-10 | `cifar10` | none |
| Vision | Beans | `beans` | none |
| Vision | Oxford-IIIT Pet | `timm/oxford-iiit-pet` | none |
| Audio | Minds14-EN | `PolyAI/minds14` | `en-US` |
| Audio | Speech Commands | `google/speech_commands` | `v0.02` |
| Audio | SUPERB Emotion Recognition | `s3prl/superb` | `er` |

You can verify that all datasets are accessible with:

```bash
python -c "from datasets import load_dataset; load_dataset('glue','sst2'); load_dataset('glue','mrpc'); load_dataset('glue','rte'); load_dataset('glue','cola')"
python -c "from datasets import load_dataset; load_dataset('GEM/e2e_nlg','default'); load_dataset('GEM/web_nlg','en')"
python -c "from datasets import load_dataset; load_dataset('cifar10'); load_dataset('beans'); load_dataset('timm/oxford-iiit-pet')"
python -c "from datasets import load_dataset; load_dataset('PolyAI/minds14','en-US'); load_dataset('google/speech_commands','v0.02'); load_dataset('s3prl/superb','er', trust_remote_code=True)"
```

Some datasets may require accepting their terms on Hugging Face or using an authenticated Hugging Face session, depending on the local environment and current dataset hosting policy.

## Regenerating Preview Files

The lightweight GLUE preview files committed in this directory can be regenerated with:

```bash
python code/reimpl/prepare_glue.py --task_name sst2 --output_dir data/glue_sst2
python code/reimpl/prepare_glue.py --task_name rte --output_dir data/glue_rte
python code/reimpl/prepare_glue.py --task_name mrpc --output_dir data/glue_mrpc
python code/reimpl/prepare_glue.py --task_name cola --output_dir data/glue_cola
```

The vision, audio, and NLG datasets are intentionally not mirrored under `data/`; the corresponding scripts in `code/vision`, `code/audio`, and `code/nlg` download and cache them automatically when first run.
