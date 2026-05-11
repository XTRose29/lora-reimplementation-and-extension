#!/bin/bash
#SBATCH -J 4782_nlu
#SBATCH -o logs/nlu/%j.out
#SBATCH -e logs/nlu/%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tx88@cornell.edu
#
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 1-00:00:00

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  SCRIPT_DIR="$SLURM_SUBMIT_DIR"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "$SCRIPT_DIR"

export KMP_DUPLICATE_LIB_OK=TRUE

mkdir -p "$SCRIPT_DIR/logs/nlu"

TASK_NAME="sst2"
MODEL_NAME="roberta-base"
RESULTS_ROOT=""
CONDA_ENV="${CONDA_ENV:-deepseek_env}"
QUICK=0
RUN_ALL=0

usage() {
  cat <<'EOF'
Usage:
  bash code/reimpl/run_task.sh
  bash code/reimpl/run_task.sh --task-name sst2 [--quick] [--model-name roberta-base]
  bash code/reimpl/run_task.sh --all [--quick] [--model-name roberta-base]
  CONDA_ENV=tx88 bash code/reimpl/run_task.sh --task-name sst2

Options:
  --task-name NAME    One of: sst2, mrpc, rte, cola. Default: sst2
  --all               Run all four tasks
  --quick             Use a smaller sample and 1 epoch for faster verification
  --model-name NAME   Hugging Face model name, default: roberta-base
  --results-root DIR  Override output root
  CONDA_ENV           Conda environment to use. Default: deepseek_env
  -h, --help          Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --task-name)
      TASK_NAME="${2:-}"
      shift 2
      ;;
    --model-name)
      MODEL_NAME="${2:-}"
      shift 2
      ;;
    --results-root)
      RESULTS_ROOT="${2:-}"
      shift 2
      ;;
    --quick)
      QUICK=1
      shift
      ;;
    --all)
      RUN_ALL=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ $RUN_ALL -eq 1 && -n "$TASK_NAME" ]]; then
  TASK_NAME=""
fi

if [[ $RUN_ALL -eq 0 && -z "$TASK_NAME" ]]; then
  echo "You must provide --task-name or --all." >&2
  exit 1
fi

if [[ $RUN_ALL -eq 1 ]]; then
  TASKS=(sst2 mrpc rte cola)
  : "${RESULTS_ROOT:=results/nlu}"
else
  case "$TASK_NAME" in
    sst2|mrpc|rte|cola) ;;
    *)
      echo "Invalid task name: $TASK_NAME" >&2
      echo "Expected one of: sst2, mrpc, rte, cola" >&2
      exit 1
      ;;
  esac
  TASKS=("$TASK_NAME")
  : "${RESULTS_ROOT:=results/nlu/$TASK_NAME}"
fi

SAMPLE_ARGS=()
EVAL_ARGS=()

if [[ $QUICK -eq 1 ]]; then
  SAMPLE_ARGS+=(--max_train_samples 1024 --max_eval_samples 512)
  EVAL_ARGS+=(--max_eval_samples 512)
fi

mkdir -p "$RESULTS_ROOT"

if [[ -f "/home/tx88/miniconda3/etc/profile.d/conda.sh" ]]; then
  # Use an existing environment with the repo dependencies installed.
  source /home/tx88/miniconda3/etc/profile.d/conda.sh
  conda activate "$CONDA_ENV"
fi

python_cmd="python"
if ! command -v "$python_cmd" >/dev/null 2>&1; then
  python_cmd="python3"
fi

run_task_method() {
  local task="$1"
  local method_name="$2"
  local method_label="$3"
  local epochs="$4"
  shift 4
  local -a extra_args=("$@")
  local output_base="$RESULTS_ROOT"
  if [[ $RUN_ALL -eq 1 ]]; then
    output_base="$RESULTS_ROOT/$task"
  fi
  local output_dir="$output_base/${task}_${method_name}"
  mkdir -p "$output_base"

  echo "Running ${method_label} on ${task}, output=${output_dir}"
  "$python_cmd" code/reimpl/train_my_lora_nlu.py \
    --task_name "$task" \
    --model_name "$MODEL_NAME" \
    --output_dir "$output_dir" \
    --epochs "$epochs" \
    --batch_size 16 \
    "${extra_args[@]}" \
    "${SAMPLE_ARGS[@]}"

  echo "Evaluating ${method_label} on ${task}"
  "$python_cmd" code/reimpl/evaluate_my_lora_nlu.py \
    --checkpoint_dir "$output_dir/checkpoint" \
    --output_dir "$output_dir" \
    "${EVAL_ARGS[@]}"
}

for task in "${TASKS[@]}"; do
  echo "Preparing GLUE task ${task}..."
  "$python_cmd" code/reimpl/prepare_glue.py \
    --task_name "$task" \
    --output_dir "data/glue_${task}" \
    --preview_samples 20
done

for task in "${TASKS[@]}"; do
  if [[ $QUICK -eq 1 ]]; then
    epochs=1
  else
    case "$task" in
      sst2) epochs=3 ;;
      mrpc|rte|cola) epochs=5 ;;
    esac
  fi

  run_task_method "$task" "ft" "FT" "$epochs" \
    --method ft --learning_rate 0.00002

  run_task_method "$task" "bitfit" "BitFit" "$epochs" \
    --method bitfit --learning_rate 0.0001

  run_task_method "$task" "adapter_0p3m" "Adapter0.3M" "$epochs" \
    --method adapter --adapter_size 16 --adapter_location output --adapter_dropout 0.0 --learning_rate 0.0002

  run_task_method "$task" "adapter_0p9m" "Adapter0.9M" "$epochs" \
    --method adapter --adapter_size 48 --adapter_location output --adapter_dropout 0.0 --learning_rate 0.0002

  run_task_method "$task" "lora" "LoRA" "$epochs" \
    --method lora --lora_r 4 --lora_alpha 32 --lora_dropout 0.1 --target_modules query,value --learning_rate 0.0002
done

if [[ $RUN_ALL -eq 1 ]]; then
  "$python_cmd" code/reimpl/plot_results.py \
    --results_root "$RESULTS_ROOT" \
    --output_dir "$RESULTS_ROOT" \
    --paper_tasks "sst2,mrpc,cola,rte"
  echo "Done. Results written to $RESULTS_ROOT"
  echo "Main table: $RESULTS_ROOT/paper_style_4task_table.md"
else
  "$python_cmd" code/reimpl/plot_results.py \
    --results_root "$RESULTS_ROOT" \
    --paper_tasks "$TASK_NAME" \
    --table_name "paper_style_${TASK_NAME}_table"
  echo "Done. Results written to $RESULTS_ROOT"
  echo "Task table: $RESULTS_ROOT/paper_style_${TASK_NAME}_table.md"
fi
