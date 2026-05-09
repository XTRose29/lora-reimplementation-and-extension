#!/bin/bash
#SBATCH -J 4782_audio
#SBATCH -o /home/tx88/4782finalproject/my_Audio/logs/%j.out
#SBATCH -e /home/tx88/4782finalproject/my_Audio/logs/%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tx88@cornell.edu
#
#SBATCH -p gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 1-00:00:00

set -euo pipefail

REPO_ROOT="/home/tx88/4782finalproject"
SCRIPT_DIR="$REPO_ROOT"
cd "$SCRIPT_DIR"

export KMP_DUPLICATE_LIB_OK=TRUE
mkdir -p "$SCRIPT_DIR/my_Audio/logs"

TASK_NAME="speech_commands"
MODEL_NAME="facebook/wav2vec2-base"
RESULTS_ROOT=""
CONDA_ENV="${CONDA_ENV:-deepseek_env}"
QUICK=0
RUN_ALL=0

usage() {
  cat <<'EOF'
Usage:
  ./my_Audio/run_audio_task.sh
  ./my_Audio/run_audio_task.sh --task-name speech_commands [--quick]
  ./my_Audio/run_audio_task.sh --all [--quick]

Options:
  --task-name NAME    One of: speech_commands, minds14_en, superb_er. Default: speech_commands
  --all               Run all three tasks
  --quick             Run a small verification subset
  --model-name NAME   Override the base audio model
  --results-root DIR  Override output root
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

if [[ $RUN_ALL -eq 1 ]]; then
  TASK_NAME=""
fi

if [[ $RUN_ALL -eq 1 ]]; then
  TASKS=(speech_commands minds14_en superb_er)
  : "${RESULTS_ROOT:=my_Audio/results/three_task_table}"
else
  case "$TASK_NAME" in
    speech_commands|minds14_en|superb_er) ;;
    *)
      echo "Invalid task name: $TASK_NAME" >&2
      exit 1
      ;;
  esac
  TASKS=("$TASK_NAME")
  : "${RESULTS_ROOT:=my_Audio/results/by_task/$TASK_NAME}"
fi

SAMPLE_ARGS=()
EVAL_ARGS=()
if [[ $QUICK -eq 1 ]]; then
  SAMPLE_ARGS+=(--max_train_samples 128 --max_eval_samples 128)
  EVAL_ARGS+=(--max_eval_samples 128)
fi

if [[ -f "/home/tx88/miniconda3/etc/profile.d/conda.sh" ]]; then
  source /home/tx88/miniconda3/etc/profile.d/conda.sh
  conda activate "$CONDA_ENV"
fi

python_cmd="python"
mkdir -p "$RESULTS_ROOT"

run_one_method() {
  local task="$1"
  local method="$2"
  local lr="$3"
  local epochs="$4"
  local output_dir="$RESULTS_ROOT/${task}_${method}"

  "$python_cmd" my_Audio/train_my_lora_audio.py \
    --task_name "$task" \
    --model_name "$MODEL_NAME" \
    --method "$method" \
    --output_dir "$output_dir" \
    --epochs "$epochs" \
    --batch_size 8 \
    --learning_rate "$lr" \
    "${SAMPLE_ARGS[@]}"

  "$python_cmd" my_Audio/evaluate_my_lora_audio.py \
    --checkpoint_dir "$output_dir/checkpoint" \
    --output_dir "$output_dir" \
    "${EVAL_ARGS[@]}"
}

for task in "${TASKS[@]}"; do
  if [[ $QUICK -eq 1 ]]; then
    epochs=1
  else
    case "$task" in
      speech_commands) epochs=3 ;;
      minds14_en|superb_er) epochs=5 ;;
    esac
  fi

  run_one_method "$task" "lora" "1e-4" "$epochs"
  run_one_method "$task" "ft" "1e-5" "$epochs"
done

"$python_cmd" my_Audio/summarize_audio_results.py \
  --results_root "$RESULTS_ROOT" \
  --table_name "summary_table"

echo "Done. Results written to $RESULTS_ROOT"
