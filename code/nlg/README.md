# NLG Experiments

This directory contains the structured natural-language generation experiments used in the final report extension.

## What It Runs

The main script, `run_qwen_nlg_generation.py`, compares full fine-tuning (FT) and LoRA with `Qwen/Qwen2.5-0.5B-Instruct` on structured NLG tasks:

- E2E NLG (`GEM/e2e_nlg`)
- WebNLG English (`GEM/web_nlg`, config `en`)
- DART (`GEM/dart`, smoke setting only in the final report)

Each task is evaluated under two prompt variants:

- `strict`: the model must produce an answer and confidence.
- `abstain`: the model is allowed to answer `I don't know`, while still reporting confidence.

The generated output format is parsed as an answer plus a self-reported confidence score. Reported metrics include BLEU, ROUGE-L, fact recall, parse rate, mean confidence, and an ECE-style proxy.

## Files

- `run_qwen_nlg_generation.py`: train/evaluate one NLG run.
- `summarize_nlg_results.py`: aggregate run directories into summary tables.
- `run_nlg_smoke_matrix.sh`: local smoke matrix for E2E, WebNLG, and DART.
- `run_nlg_full_matrix.sh`: larger E2E/WebNLG matrix used for the final report.
- `make_nlg_smoke_html.py`: utility for viewing sampled generated outputs.
- `cluster/*.sub`: SLURM submission scripts used on the course cluster.

## Example Commands

From the repository root:

```bash
export PYTHONPATH="$PWD/code:$PYTHONPATH"
bash code/nlg/run_nlg_smoke_matrix.sh
```

For the larger reported run:

```bash
bash code/nlg/run_nlg_full_matrix.sh
```

On a SLURM cluster, submit one task at a time, for example:

```bash
sbatch code/nlg/cluster/RUN_nlg_full_e2e.sub
sbatch code/nlg/cluster/RUN_nlg_full_webnlg.sub
```

The full runs are GPU-intensive. The DART full run was not included in the final report because the available cluster allocation hit the wall-time limit; DART is included as a smoke-scale result.
