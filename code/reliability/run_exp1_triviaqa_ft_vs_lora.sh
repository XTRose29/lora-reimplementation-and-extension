#!/bin/bash
#SBATCH -J triviaqa_ft_lora
#SBATCH -o /home/tx88/4782finalproject/cola_reliability/logs/%j.out
#SBATCH -e /home/tx88/4782finalproject/cola_reliability/logs/%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tx88@cornell.edu
#
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH -t 2-00:00:00

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
RESULTS_ROOT="cola_reliability/results/exp1_triviaqa_ft_vs_lora"
ROBERTA_MODEL="roberta-base"
QWEN_MODEL="Qwen/Qwen2.5-0.5B-Instruct"
DATASET_NAME="trivia_qa"
DATASET_CONFIG="rc.nocontext"
SOURCE_SPLIT="validation"
QUICK=0
EPOCHS=3
BATCH_SIZE=8
MAX_TRAIN_QUESTIONS=1000
MAX_EVAL_QUESTIONS=1000
NUM_NEGATIVES=3
ABSTENTION_THRESHOLD=0.80
CALIBRATION_BINS=15
SEED=42
DEVICE=""
LORA_R=4
LORA_ALPHA=32
LORA_DROPOUT=0.1
LORA_PLACEMENT="attention"
ATTENTION_TARGETS="auto"

usage() {
  cat <<'EOF'
Usage:
  ./cola_reliability/run_exp1_triviaqa_ft_vs_lora.sh
  ./cola_reliability/run_exp1_triviaqa_ft_vs_lora.sh --quick --device cpu
  sbatch cola_reliability/run_exp1_triviaqa_ft_vs_lora.sh

Experiment:
  Uses TriviaQA validation as a small answer-ranking dataset. Each question has
  one gold answer candidate and sampled negative answer candidates. Runs:
  roberta-base FT, roberta-base LoRA, Qwen-0.5B FT, Qwen-0.5B LoRA.

Options:
  --quick                    1 epoch, 128 train questions, 128 eval questions
  --results-root DIR         Default: cola_reliability/results/exp1_triviaqa_ft_vs_lora
  --roberta-model NAME       Default: roberta-base
  --qwen-model NAME          Default: Qwen/Qwen2.5-0.5B-Instruct
  --dataset-name NAME        Default: trivia_qa
  --dataset-config NAME      Default: rc.nocontext
  --source-split NAME        Default: validation
  --epochs N                 Default: 3
  --batch-size N             Default: 8
  --max-train-questions N    Default: 1000
  --max-eval-questions N     Default: 1000
  --num-negatives N          Default: 3
  --threshold FLOAT          Abstention threshold. Default: 0.80
  --calibration-bins N       ECE bins. Default: 15
  --seed N                   Default: 42
  --device DEVICE            Optional: cpu, cuda, cuda:0
  --lora-r N                 Default: 4
  --lora-alpha FLOAT         Default: 32
  --lora-dropout FLOAT       Default: 0.1
  --lora-placement NAME      attention, mlp, attention_mlp. Default: attention
  --attention-targets CSV    Default: auto
  CONDA_ENV                  Conda environment. Default: deepseek_env
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --quick) QUICK=1; shift ;;
    --results-root) RESULTS_ROOT="${2:-}"; shift 2 ;;
    --roberta-model) ROBERTA_MODEL="${2:-}"; shift 2 ;;
    --qwen-model) QWEN_MODEL="${2:-}"; shift 2 ;;
    --dataset-name) DATASET_NAME="${2:-}"; shift 2 ;;
    --dataset-config) DATASET_CONFIG="${2:-}"; shift 2 ;;
    --source-split) SOURCE_SPLIT="${2:-}"; shift 2 ;;
    --epochs) EPOCHS="${2:-}"; shift 2 ;;
    --batch-size) BATCH_SIZE="${2:-}"; shift 2 ;;
    --max-train-questions) MAX_TRAIN_QUESTIONS="${2:-}"; shift 2 ;;
    --max-eval-questions) MAX_EVAL_QUESTIONS="${2:-}"; shift 2 ;;
    --num-negatives) NUM_NEGATIVES="${2:-}"; shift 2 ;;
    --threshold) ABSTENTION_THRESHOLD="${2:-}"; shift 2 ;;
    --calibration-bins) CALIBRATION_BINS="${2:-}"; shift 2 ;;
    --seed) SEED="${2:-}"; shift 2 ;;
    --device) DEVICE="${2:-}"; shift 2 ;;
    --lora-r) LORA_R="${2:-}"; shift 2 ;;
    --lora-alpha) LORA_ALPHA="${2:-}"; shift 2 ;;
    --lora-dropout) LORA_DROPOUT="${2:-}"; shift 2 ;;
    --lora-placement) LORA_PLACEMENT="${2:-}"; shift 2 ;;
    --attention-targets) ATTENTION_TARGETS="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
done

if [[ $QUICK -eq 1 ]]; then
  EPOCHS=1
  BATCH_SIZE=4
  MAX_TRAIN_QUESTIONS=128
  MAX_EVAL_QUESTIONS=128
  RESULTS_ROOT="cola_reliability/results/quick_exp1_triviaqa_ft_vs_lora"
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

device_args=()
if [[ -n "$DEVICE" ]]; then
  device_args+=(--device "$DEVICE")
fi

common_args=(
  --dataset_name "$DATASET_NAME"
  --dataset_config "$DATASET_CONFIG"
  --source_split "$SOURCE_SPLIT"
  --epochs "$EPOCHS"
  --batch_size "$BATCH_SIZE"
  --max_train_questions "$MAX_TRAIN_QUESTIONS"
  --max_eval_questions "$MAX_EVAL_QUESTIONS"
  --num_negative_candidates "$NUM_NEGATIVES"
  --abstention_threshold "$ABSTENTION_THRESHOLD"
  --calibration_bins "$CALIBRATION_BINS"
  --seed "$SEED"
  "${device_args[@]}"
)

settings_csv="$RESULTS_ROOT/settings_table.csv"
settings_md="$RESULTS_ROOT/settings_table.md"
printf "task_name,method,model_name,epochs,batch_size,max_train_questions,max_eval_questions,num_negative_candidates,lora_r,lora_alpha,lora_dropout,lora_placement,attention_targets,abstention_threshold,calibration_bins,seed,output_dir\n" > "$settings_csv"
{
  printf "| task_name | method | model_name | epochs | batch_size | max_train_questions | max_eval_questions | num_negative_candidates | lora_r | lora_alpha | lora_dropout | lora_placement | attention_targets | abstention_threshold | calibration_bins | seed | output_dir |\n"
  printf "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
} > "$settings_md"

run_method() {
  local model_label="$1"
  local model_name="$2"
  local method="$3"
  local output_dir="$RESULTS_ROOT/${model_label}_${method}"

  if [[ "$method" == "ft" ]]; then
    printf "triviaqa,ft,%s,%s,%s,%s,%s,%s,,,,,,%s,%s,%s,%s\n" "$model_name" "$EPOCHS" "$BATCH_SIZE" "$MAX_TRAIN_QUESTIONS" "$MAX_EVAL_QUESTIONS" "$NUM_NEGATIVES" "$ABSTENTION_THRESHOLD" "$CALIBRATION_BINS" "$SEED" "$output_dir" >> "$settings_csv"
    printf "| triviaqa | ft | %s | %s | %s | %s | %s | %s |  |  |  |  |  | %s | %s | %s | %s |\n" "$model_name" "$EPOCHS" "$BATCH_SIZE" "$MAX_TRAIN_QUESTIONS" "$MAX_EVAL_QUESTIONS" "$NUM_NEGATIVES" "$ABSTENTION_THRESHOLD" "$CALIBRATION_BINS" "$SEED" "$output_dir" >> "$settings_md"
    "$python_cmd" cola_reliability/run_triviaqa_reliability.py \
      --model_name "$model_name" \
      --method ft \
      --output_dir "$output_dir" \
      "${common_args[@]}"
  else
    printf "triviaqa,lora,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" "$model_name" "$EPOCHS" "$BATCH_SIZE" "$MAX_TRAIN_QUESTIONS" "$MAX_EVAL_QUESTIONS" "$NUM_NEGATIVES" "$LORA_R" "$LORA_ALPHA" "$LORA_DROPOUT" "$LORA_PLACEMENT" "$ATTENTION_TARGETS" "$ABSTENTION_THRESHOLD" "$CALIBRATION_BINS" "$SEED" "$output_dir" >> "$settings_csv"
    printf "| triviaqa | lora | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |\n" "$model_name" "$EPOCHS" "$BATCH_SIZE" "$MAX_TRAIN_QUESTIONS" "$MAX_EVAL_QUESTIONS" "$NUM_NEGATIVES" "$LORA_R" "$LORA_ALPHA" "$LORA_DROPOUT" "$LORA_PLACEMENT" "$ATTENTION_TARGETS" "$ABSTENTION_THRESHOLD" "$CALIBRATION_BINS" "$SEED" "$output_dir" >> "$settings_md"
    "$python_cmd" cola_reliability/run_triviaqa_reliability.py \
      --model_name "$model_name" \
      --method lora \
      --output_dir "$output_dir" \
      --lora_r "$LORA_R" \
      --lora_alpha "$LORA_ALPHA" \
      --lora_dropout "$LORA_DROPOUT" \
      --lora_placement "$LORA_PLACEMENT" \
      --attention_targets "$ATTENTION_TARGETS" \
      "${common_args[@]}"
  fi
}

echo "TriviaQA Exp1: RoBERTa-base FT vs LoRA"
run_method "roberta_base" "$ROBERTA_MODEL" "ft"
run_method "roberta_base" "$ROBERTA_MODEL" "lora"

echo "TriviaQA Exp1: Qwen 0.5B FT vs LoRA"
run_method "qwen_0p5b" "$QWEN_MODEL" "ft"
run_method "qwen_0p5b" "$QWEN_MODEL" "lora"

"$python_cmd" cola_reliability/summarize_results.py --results_root "$RESULTS_ROOT"

echo "Done. TriviaQA Exp1 results written to $RESULTS_ROOT"
echo "Settings table: $RESULTS_ROOT/settings_table.md"
echo "Result table: $RESULTS_ROOT/summary_table.md"
