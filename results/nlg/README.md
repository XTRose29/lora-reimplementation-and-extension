# NLG Results

This directory contains the natural-language generation outputs used for the final report's NLG extension.

## Layout

- `full/e2e/`: E2E full-scale summaries and downloaded run artifacts.
- `full/webnlg/`: WebNLG full-scale summaries and downloaded run artifacts.
- `paper_gpt2_full/e2e/`: GPT-2 Medium FT/LoRA E2E reproduction summaries and prediction files.
- `paper_gpt2_full/webnlg/`: GPT-2 Medium FT/LoRA WebNLG extension summaries and prediction files.
- `smoke/e2e/`: E2E smoke summary tables.
- `smoke/webnlg/`: WebNLG smoke summary tables.
- `smoke/dart/`: DART smoke summary tables.

Compact report-level tables are also stored at `../summary_tables/nlg_full.csv` and `../summary_tables/nlg_gpt2_e2e_reproduction.csv`.

## Metrics

- `BLEU`: n-gram overlap with reference text.
- `ROUGE-L`: longest-sequence overlap with reference text.
- `Fact Recall`: fraction of structured input facts expressed in the generated text.
- `Parse Rate`: fraction of outputs that follow the required answer/confidence format.
- `Mean Conf.`: average self-reported confidence.
- `ECE Proxy`: absolute gap between mean confidence and fact recall in the report table.

## Notes

The final report uses full-scale E2E and WebNLG results with 5,000 training examples and 1,000 evaluation examples. DART is included at smoke scale because the full DART cluster job exceeded the available wall-time budget.

The GPT-2 Medium E2E reproduction table compares only BLEU and ROUGE-L because those are the metrics shared by the LoRA paper table and our evaluation pipeline.

Run scripts are in `../../code/nlg/`.
