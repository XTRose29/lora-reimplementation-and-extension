#!/bin/bash
#SBATCH -J cola_reliability
#SBATCH -o /home/tx88/4782finalproject/cola_reliability/logs/%j.out
#SBATCH -e /home/tx88/4782finalproject/cola_reliability/logs/%j.err
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
  REPO_ROOT="$SLURM_SUBMIT_DIR"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi
cd "$REPO_ROOT"

export KMP_DUPLICATE_LIB_OK=TRUE

CONDA_ENV="${CONDA_ENV:-deepseek_env}"
MODEL_NAME="roberta-base"
RESULTS_ROOT="cola_reliability/results/ft_vs_lora"
QUICK=0
EPOCHS=5
BATCH_SIZE=16
TRAIN_SAMPLES=""
EVAL_SAMPLES=""
OOD_SAMPLES=1000
ABSTENTION_THRESHOLD=0.80
LORA_R=4
LORA_PLACEMENT="attention"
SEED=42

usage() {
  cat <<'EOF'
Usage:
  ./cola_reliability/run_task.sh
  ./cola_reliability/run_task.sh --quick
  sbatch cola_reliability/run_task.sh --quick
  CONDA_ENV=tx88 ./cola_reliability/run_task.sh --epochs 5 --results-root cola_reliability/results/ft_vs_lora

Options:
  --quick                  Run a 1-epoch smoke comparison on 128 examples
  --model-name NAME        Hugging Face model name. Default: roberta-base
  --results-root DIR       Output root. Default: cola_reliability/results/ft_vs_lora
  --epochs N               Number of epochs for both FT and LoRA. Default: 5
  --batch-size N           Batch size. Default: 16
  --max-train-samples N    Limit CoLA train examples
  --max-eval-samples N     Limit CoLA validation examples
  --max-ood-samples N      Limit SST-2 OOD examples. Default: 1000
  --threshold FLOAT        Abstention threshold. Default: 0.80
  --lora-r N               LoRA rank. Default: 4
  --lora-placement NAME    One of: attention, mlp, attention_mlp. Default: attention
  --seed N                 Random seed. Default: 42
  CONDA_ENV                Conda environment to use. Default: deepseek_env
  -h, --help               Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --quick)
      QUICK=1
      shift
      ;;
    --model-name)
      MODEL_NAME="${2:-}"
      shift 2
      ;;
    --results-root)
      RESULTS_ROOT="${2:-}"
      shift 2
      ;;
    --epochs)
      EPOCHS="${2:-}"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="${2:-}"
      shift 2
      ;;
    --max-train-samples)
      TRAIN_SAMPLES="${2:-}"
      shift 2
      ;;
    --max-eval-samples)
      EVAL_SAMPLES="${2:-}"
      shift 2
      ;;
    --max-ood-samples)
      OOD_SAMPLES="${2:-}"
      shift 2
      ;;
    --threshold)
      ABSTENTION_THRESHOLD="${2:-}"
      shift 2
      ;;
    --lora-r)
      LORA_R="${2:-}"
      shift 2
      ;;
    --lora-placement)
      LORA_PLACEMENT="${2:-}"
      shift 2
      ;;
    --seed)
      SEED="${2:-}"
      shift 2
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

case "$LORA_PLACEMENT" in
  attention|mlp|attention_mlp) ;;
  *)
    echo "Invalid --lora-placement: $LORA_PLACEMENT" >&2
    echo "Expected one of: attention, mlp, attention_mlp" >&2
    exit 1
    ;;
esac

if [[ $QUICK -eq 1 ]]; then
  EPOCHS=1
  BATCH_SIZE=8
  TRAIN_SAMPLES=128
  EVAL_SAMPLES=128
  OOD_SAMPLES=128
  RESULTS_ROOT="cola_reliability/results/quick_ft_vs_lora"
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

sample_args=()
if [[ -n "$TRAIN_SAMPLES" ]]; then
  sample_args+=(--max_train_samples "$TRAIN_SAMPLES")
fi
if [[ -n "$EVAL_SAMPLES" ]]; then
  sample_args+=(--max_eval_samples "$EVAL_SAMPLES")
fi
if [[ -n "$OOD_SAMPLES" ]]; then
  sample_args+=(--max_ood_samples "$OOD_SAMPLES")
fi

common_args=(
  --model_name "$MODEL_NAME"
  --epochs "$EPOCHS"
  --batch_size "$BATCH_SIZE"
  --abstention_threshold "$ABSTENTION_THRESHOLD"
  --seed "$SEED"
  "${sample_args[@]}"
)

echo "Running CoLA full fine-tuning reliability baseline..."
"$python_cmd" cola_reliability/run_cola_reliability.py \
  --method ft \
  --output_dir "$RESULTS_ROOT/ft" \
  "${common_args[@]}"

echo "Running CoLA LoRA reliability comparison..."
"$python_cmd" cola_reliability/run_cola_reliability.py \
  --method lora \
  --output_dir "$RESULTS_ROOT/lora_r${LORA_R}_${LORA_PLACEMENT}" \
  --lora_r "$LORA_R" \
  --lora_placement "$LORA_PLACEMENT" \
  "${common_args[@]}"

"$python_cmd" cola_reliability/summarize_results.py --results_root "$RESULTS_ROOT"

echo "Done. Results written to $RESULTS_ROOT"
echo "Comparison table: $RESULTS_ROOT/summary_table.md"
