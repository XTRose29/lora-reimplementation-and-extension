#!/bin/bash
#SBATCH -J glue_exp1_ft_lora
#SBATCH -o logs/reliability/%j.out
#SBATCH -e logs/reliability/%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tx88@cornell.edu
#
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 2-00:00:00

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  REPO_ROOT="$SLURM_SUBMIT_DIR"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
fi
cd "$REPO_ROOT"

export KMP_DUPLICATE_LIB_OK=TRUE

CONDA_ENV="${CONDA_ENV:-deepseek_env}"
MODEL_NAME="roberta-base"
RESULTS_ROOT="results/reliability"
TASKS_CSV="cola,mrpc,rte,sst2"
QUICK=0
EPOCHS=5
BATCH_SIZE=16
TRAIN_SAMPLES=""
EVAL_SAMPLES=""
ABSTENTION_THRESHOLD=0.80
CALIBRATION_BINS=15
SEED=42
DEVICE=""
LORA_R=4
LORA_ALPHA=32
LORA_DROPOUT=0.1
LORA_PLACEMENT="attention"

usage() {
  cat <<'EOF'
Usage:
  bash code/reliability/run_exp1_ft_vs_lora.sh
  bash code/reliability/run_exp1_ft_vs_lora.sh --quick
  sbatch code/reliability/run_exp1_ft_vs_lora.sh

Options:
  --quick                  Run 1 epoch on 128 train/eval examples per task
  --tasks CSV              Default: cola,mrpc,rte,sst2
  --model-name NAME        Default: roberta-base
  --results-root DIR       Default: results/reliability
  --epochs N               Default: 5
  --batch-size N           Default: 16
  --max-train-samples N    Limit train examples per task
  --max-eval-samples N     Limit validation examples per task
  --threshold FLOAT        Abstention threshold. Default: 0.80
  --calibration-bins N     ECE bins. Default: 15
  --seed N                 Default: 42
  --device DEVICE          Optional: cpu, cuda, cuda:0
  --lora-r N               Default: 4
  --lora-alpha FLOAT       Default: 32
  --lora-dropout FLOAT     Default: 0.1
  --lora-placement NAME    attention, mlp, or attention_mlp. Default: attention
  CONDA_ENV                Conda environment. Default: deepseek_env
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --quick) QUICK=1; shift ;;
    --tasks) TASKS_CSV="${2:-}"; shift 2 ;;
    --model-name) MODEL_NAME="${2:-}"; shift 2 ;;
    --results-root) RESULTS_ROOT="${2:-}"; shift 2 ;;
    --epochs) EPOCHS="${2:-}"; shift 2 ;;
    --batch-size) BATCH_SIZE="${2:-}"; shift 2 ;;
    --max-train-samples) TRAIN_SAMPLES="${2:-}"; shift 2 ;;
    --max-eval-samples) EVAL_SAMPLES="${2:-}"; shift 2 ;;
    --threshold) ABSTENTION_THRESHOLD="${2:-}"; shift 2 ;;
    --calibration-bins) CALIBRATION_BINS="${2:-}"; shift 2 ;;
    --seed) SEED="${2:-}"; shift 2 ;;
    --device) DEVICE="${2:-}"; shift 2 ;;
    --lora-r) LORA_R="${2:-}"; shift 2 ;;
    --lora-alpha) LORA_ALPHA="${2:-}"; shift 2 ;;
    --lora-dropout) LORA_DROPOUT="${2:-}"; shift 2 ;;
    --lora-placement) LORA_PLACEMENT="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
done

if [[ $QUICK -eq 1 ]]; then
  EPOCHS=1
  BATCH_SIZE=8
  TRAIN_SAMPLES=128
  EVAL_SAMPLES=128
  RESULTS_ROOT="results/reliability/quick"
fi

mkdir -p "$RESULTS_ROOT" "logs/reliability"

if [[ -f "/home/tx88/miniconda3/etc/profile.d/conda.sh" ]]; then
  source /home/tx88/miniconda3/etc/profile.d/conda.sh
  conda activate "$CONDA_ENV"
fi

python_cmd="python"
if ! command -v "$python_cmd" >/dev/null 2>&1; then
  python_cmd="python3"
fi

IFS=',' read -r -a TASKS <<< "$TASKS_CSV"

sample_args=()
if [[ -n "$TRAIN_SAMPLES" ]]; then
  sample_args+=(--max_train_samples "$TRAIN_SAMPLES")
fi
if [[ -n "$EVAL_SAMPLES" ]]; then
  sample_args+=(--max_eval_samples "$EVAL_SAMPLES")
fi
if [[ -n "$DEVICE" ]]; then
  sample_args+=(--device "$DEVICE")
fi

common_args=(
  --model_name "$MODEL_NAME"
  --epochs "$EPOCHS"
  --batch_size "$BATCH_SIZE"
  --abstention_threshold "$ABSTENTION_THRESHOLD"
  --calibration_bins "$CALIBRATION_BINS"
  --seed "$SEED"
  --ood_task none
  "${sample_args[@]}"
)

settings_csv="$RESULTS_ROOT/settings_table.csv"
settings_md="$RESULTS_ROOT/settings_table.md"
printf "task_name,method,model_name,epochs,batch_size,max_train_samples,max_eval_samples,lora_r,lora_alpha,lora_dropout,lora_placement,abstention_threshold,calibration_bins,seed,output_dir\n" > "$settings_csv"
{
  printf "| task_name | method | model_name | epochs | batch_size | max_train_samples | max_eval_samples | lora_r | lora_alpha | lora_dropout | lora_placement | abstention_threshold | calibration_bins | seed | output_dir |\n"
  printf "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
} > "$settings_md"

for task in "${TASKS[@]}"; do
  task="${task// /}"
  ft_dir="$RESULTS_ROOT/${task}_ft"
  lora_dir="$RESULTS_ROOT/${task}_lora_r${LORA_R}_${LORA_PLACEMENT}"
  printf "%s,ft,%s,%s,%s,%s,%s,,,,,%s,%s,%s,%s\n" "$task" "$MODEL_NAME" "$EPOCHS" "$BATCH_SIZE" "$TRAIN_SAMPLES" "$EVAL_SAMPLES" "$ABSTENTION_THRESHOLD" "$CALIBRATION_BINS" "$SEED" "$ft_dir" >> "$settings_csv"
  printf "%s,lora,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" "$task" "$MODEL_NAME" "$EPOCHS" "$BATCH_SIZE" "$TRAIN_SAMPLES" "$EVAL_SAMPLES" "$LORA_R" "$LORA_ALPHA" "$LORA_DROPOUT" "$LORA_PLACEMENT" "$ABSTENTION_THRESHOLD" "$CALIBRATION_BINS" "$SEED" "$lora_dir" >> "$settings_csv"
  printf "| %s | ft | %s | %s | %s | %s | %s |  |  |  |  | %s | %s | %s | %s |\n" "$task" "$MODEL_NAME" "$EPOCHS" "$BATCH_SIZE" "$TRAIN_SAMPLES" "$EVAL_SAMPLES" "$ABSTENTION_THRESHOLD" "$CALIBRATION_BINS" "$SEED" "$ft_dir" >> "$settings_md"
  printf "| %s | lora | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |\n" "$task" "$MODEL_NAME" "$EPOCHS" "$BATCH_SIZE" "$TRAIN_SAMPLES" "$EVAL_SAMPLES" "$LORA_R" "$LORA_ALPHA" "$LORA_DROPOUT" "$LORA_PLACEMENT" "$ABSTENTION_THRESHOLD" "$CALIBRATION_BINS" "$SEED" "$lora_dir" >> "$settings_md"

  echo "Exp1: ${task} full fine-tuning"
  "$python_cmd" code/reliability/run_cola_reliability.py \
    --task_name "$task" \
    --method ft \
    --output_dir "$ft_dir" \
    "${common_args[@]}"

  echo "Exp1: ${task} LoRA"
  "$python_cmd" code/reliability/run_cola_reliability.py \
    --task_name "$task" \
    --method lora \
    --output_dir "$lora_dir" \
    --lora_r "$LORA_R" \
    --lora_alpha "$LORA_ALPHA" \
    --lora_dropout "$LORA_DROPOUT" \
    --lora_placement "$LORA_PLACEMENT" \
    "${common_args[@]}"
done

"$python_cmd" code/reliability/summarize_results.py --results_root "$RESULTS_ROOT"

echo "Done. Exp1 results written to $RESULTS_ROOT"
echo "Table: $RESULTS_ROOT/summary_table.md"
