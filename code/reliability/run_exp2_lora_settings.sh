#!/bin/bash
#SBATCH -J glue_exp2_lora
#SBATCH -o /home/tx88/4782finalproject/cola_reliability/logs/%j.out
#SBATCH -e /home/tx88/4782finalproject/cola_reliability/logs/%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tx88@cornell.edu
#
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 4-00:00:00

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  REPO_ROOT="$SLURM_SUBMIT_DIR"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi
cd "$REPO_ROOT"

export KMP_DUPLICATE_LIB_OK=TRUE

CONDA_ENV="${CONDA_ENV:-deepseek_env}"
MODEL_NAME="roberta-base"
RESULTS_ROOT="cola_reliability/results/exp2_lora_settings"
TASKS_CSV="cola,mrpc,rte,sst2"
RANKS_CSV="1,4,8,16"
PLACEMENTS_CSV="attention,mlp,attention_mlp"
QUICK=0
EPOCHS=5
BATCH_SIZE=16
TRAIN_SAMPLES=""
EVAL_SAMPLES=""
ABSTENTION_THRESHOLD=0.80
CALIBRATION_BINS=15
SEED=42
DEVICE=""
LORA_ALPHA=32
LORA_DROPOUT=0.1

usage() {
  cat <<'EOF'
Usage:
  ./cola_reliability/run_exp2_lora_settings.sh
  ./cola_reliability/run_exp2_lora_settings.sh --quick
  sbatch cola_reliability/run_exp2_lora_settings.sh

Options:
  --quick                  Run rank=4, placement=attention, 1 epoch, 128 examples
  --tasks CSV              Default: cola,mrpc,rte,sst2
  --ranks CSV              Default: 1,4,8,16
  --placements CSV         Default: attention,mlp,attention_mlp
  --model-name NAME        Default: roberta-base
  --results-root DIR       Default: cola_reliability/results/exp2_lora_settings
  --epochs N               Default: 5
  --batch-size N           Default: 16
  --max-train-samples N    Limit train examples per task
  --max-eval-samples N     Limit validation examples per task
  --threshold FLOAT        Abstention threshold. Default: 0.80
  --calibration-bins N     ECE bins. Default: 15
  --seed N                 Default: 42
  --device DEVICE          Optional: cpu, cuda, cuda:0
  --lora-alpha FLOAT       Default: 32
  --lora-dropout FLOAT     Default: 0.1
  CONDA_ENV                Conda environment. Default: deepseek_env
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --quick) QUICK=1; shift ;;
    --tasks) TASKS_CSV="${2:-}"; shift 2 ;;
    --ranks) RANKS_CSV="${2:-}"; shift 2 ;;
    --placements) PLACEMENTS_CSV="${2:-}"; shift 2 ;;
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
    --lora-alpha) LORA_ALPHA="${2:-}"; shift 2 ;;
    --lora-dropout) LORA_DROPOUT="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
done

if [[ $QUICK -eq 1 ]]; then
  EPOCHS=1
  BATCH_SIZE=8
  TRAIN_SAMPLES=128
  EVAL_SAMPLES=128
  RANKS_CSV="4"
  PLACEMENTS_CSV="attention"
  RESULTS_ROOT="cola_reliability/results/quick_exp2_lora_settings"
fi

mkdir -p "$RESULTS_ROOT" "cola_reliability/logs"

if [[ -f "/home/tx88/miniconda3/etc/profile.d/conda.sh" ]]; then
  source /home/tx88/miniconda3/etc/profile.d/conda.sh
  conda activate "$CONDA_ENV"
fi

python_cmd="python"
if ! command -v "$python_cmd" >/dev/null 2>&1; then
  python_cmd="python3"
fi

IFS=',' read -r -a TASKS <<< "$TASKS_CSV"
IFS=',' read -r -a RANKS <<< "$RANKS_CSV"
IFS=',' read -r -a PLACEMENTS <<< "$PLACEMENTS_CSV"

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
  --method lora
  --epochs "$EPOCHS"
  --batch_size "$BATCH_SIZE"
  --abstention_threshold "$ABSTENTION_THRESHOLD"
  --calibration_bins "$CALIBRATION_BINS"
  --seed "$SEED"
  --lora_alpha "$LORA_ALPHA"
  --lora_dropout "$LORA_DROPOUT"
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
  for rank in "${RANKS[@]}"; do
    rank="${rank// /}"
    for placement in "${PLACEMENTS[@]}"; do
      placement="${placement// /}"
      output_dir="$RESULTS_ROOT/${task}_lora_r${rank}_${placement}"
      printf "%s,lora,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" "$task" "$MODEL_NAME" "$EPOCHS" "$BATCH_SIZE" "$TRAIN_SAMPLES" "$EVAL_SAMPLES" "$rank" "$LORA_ALPHA" "$LORA_DROPOUT" "$placement" "$ABSTENTION_THRESHOLD" "$CALIBRATION_BINS" "$SEED" "$output_dir" >> "$settings_csv"
      printf "| %s | lora | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |\n" "$task" "$MODEL_NAME" "$EPOCHS" "$BATCH_SIZE" "$TRAIN_SAMPLES" "$EVAL_SAMPLES" "$rank" "$LORA_ALPHA" "$LORA_DROPOUT" "$placement" "$ABSTENTION_THRESHOLD" "$CALIBRATION_BINS" "$SEED" "$output_dir" >> "$settings_md"

      echo "Exp2: ${task} LoRA r=${rank} placement=${placement}"
      "$python_cmd" cola_reliability/run_cola_reliability.py \
        --task_name "$task" \
        --output_dir "$output_dir" \
        --lora_r "$rank" \
        --lora_placement "$placement" \
        "${common_args[@]}"
    done
  done
done

"$python_cmd" cola_reliability/summarize_results.py --results_root "$RESULTS_ROOT"

echo "Done. Exp2 results written to $RESULTS_ROOT"
echo "Table: $RESULTS_ROOT/summary_table.md"
