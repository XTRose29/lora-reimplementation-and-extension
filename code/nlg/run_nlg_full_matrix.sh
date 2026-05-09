#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

TASK="${1:?usage: run_nlg_full_matrix.sh <e2e|webnlg|dart>}"
ROOT="results/nlg/full/${TASK}"

mkdir -p "${ROOT}"

python code/nlg/run_qwen_nlg_generation.py \
  --task "${TASK}" \
  --method ft \
  --prompt_variants strict abstain \
  --output_dir "${ROOT}/ft_eval_bundle" \
  --max_train_examples 5000 \
  --max_eval_examples 1000 \
  --epochs 2 \
  --batch_size 2 \
  --max_length 384 \
  --max_new_tokens 64 \
  --seed 42

python code/nlg/run_qwen_nlg_generation.py \
  --task "${TASK}" \
  --method lora \
  --prompt_variants strict abstain \
  --output_dir "${ROOT}/lora_eval_bundle" \
  --max_train_examples 5000 \
  --max_eval_examples 1000 \
  --epochs 2 \
  --batch_size 2 \
  --max_length 384 \
  --max_new_tokens 64 \
  --lora_r 4 \
  --lora_alpha 32 \
  --lora_dropout 0.1 \
  --seed 42

python code/nlg/summarize_nlg_results.py --results_root "${ROOT}"
