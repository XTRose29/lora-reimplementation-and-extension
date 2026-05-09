#!/bin/bash
set -euo pipefail

TASK="${1:?usage: run_nlg_gpt2_full_ft_matrix.sh <e2e|webnlg>}"
MODEL_NAME="${MODEL_NAME:-gpt2-medium}"
MODEL_TAG="$(echo "${MODEL_NAME}" | tr '/.-' '___')"
ROOT="nlg/results/paper_gpt2_full/${TASK}"

mkdir -p "${ROOT}"

python nlg/run_qwen_nlg_generation.py \
  --task "${TASK}" \
  --model_name "${MODEL_NAME}" \
  --method ft \
  --prompt_variant strict \
  --output_dir "${ROOT}/${MODEL_TAG}_${TASK}_ft" \
  --max_train_examples 5000 \
  --max_eval_examples 1000 \
  --epochs 2 \
  --batch_size 2 \
  --learning_rate 5e-5 \
  --max_length 384 \
  --max_new_tokens 64 \
  --seed 42

python nlg/summarize_nlg_results.py --results_root "${ROOT}"
