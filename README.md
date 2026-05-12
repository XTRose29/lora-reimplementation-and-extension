# A New Look at the "Battle" between LoRA and Full Fine-Tuning

**Authors:** Tianruo Rose Xu and Weijie Zhou  
**Course:** CS 4782 final project, Cornell University

## 1. Introduction

This repository contains our re-implementation and extension project for **Hu et al. (2022), "LoRA: Low-Rank Adaptation of Large Language Models."** LoRA's main contribution is to adapt large pretrained models by freezing the backbone and training low-rank update matrices, greatly reducing trainable parameters.

Our goal is to test whether LoRA behaves like a parameter-efficient substitute for full fine-tuning (FT), or whether the low-rank constraint changes downstream performance and reliability.

## 2. Chosen Result

We aimed to reproduce the paper's central efficiency-performance claim: LoRA can match or exceed FT while updating far fewer parameters. The key original references are **Table 2** for RoBERTa on GLUE and **Table 3** for GPT-2 Medium/Large on E2E NLG.

The LoRA update used throughout the project is:

```text
W = W0 + Delta W,     Delta W = (alpha / r) B A
```

## 3. GitHub Contents

```text
code/       training, evaluation, LoRA modules, and summarizers
data/       lightweight GLUE previews and dataset download instructions
results/    metrics, configs, logs, and predictions used in the final report
report/     final project report PDF
poster/     final poster PDF
```

The main implementation folders are `code/reimpl` for RoBERTa GLUE, `code/nlg` for GPT-2/Qwen generation, `code/vision` for ViT image classification, `code/audio` for wav2vec2 speech classification, and `code/reliability` for calibration metrics.

## 4. Re-implementation Details

We implemented FT, LoRA, adapters, and BitFit for RoBERTa-base on GLUE, then extended the FT-vs-LoRA comparison to GPT-2 Medium on E2E, ViT-base on vision tasks, wav2vec2-base on audio tasks, and Qwen2.5-0.5B-Instruct for confidence-aware NLG.

Datasets include SST-2, MRPC, RTE, CoLA, E2E, WebNLG, CIFAR-10, Beans, Oxford-IIIT Pet, Minds14-EN, Speech Commands, and SUPERB ER. Metrics include accuracy, F1, Matthews correlation, BLEU, ROUGE-L, fact recall, and expected calibration error (ECE).

## 5. Reproduction Steps

**Compute notice.** Full reproduction needs a CUDA-capable GPU and enough disk space for Hugging Face model/dataset caches. We ran the full experiments on a single NVIDIA RTX A6000 GPU. GPT-2 Medium FT/LoRA NLG runs took roughly 0.5--2.5 hours per run, while RoBERTa-base NLU runs were usually shorter, roughly 0.5--2 hours per run. CPU runs are useful only for small checks or smoke tests.

Create an environment from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r code/requirements.txt
export PYTHONPATH="$PWD/code:$PYTHONPATH"
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r code\requirements.txt
$env:PYTHONPATH = "$PWD\code;$env:PYTHONPATH"
```

Example smoke runs:

```bash
python code/reimpl/train_my_lora_nlu.py --task_name sst2 --method lora --output_dir results/nlu/example_sst2_lora --max_train_samples 128 --max_eval_samples 128 --epochs 1
python code/vision/train_my_lora_vision.py --task_name cifar10 --method lora --output_dir results/vision/example_cifar10_lora --max_train_samples 128 --max_eval_samples 128 --epochs 1
python code/audio/train_my_lora_audio.py --task_name speech_commands --method lora --output_dir results/audio/example_speech_commands_lora --max_train_samples 128 --max_eval_samples 128 --epochs 1
python code/reliability/run_cola_reliability.py --task_name cola --method lora --output_dir results/reliability/example_cola_lora --max_train_samples 128 --max_eval_samples 128 --epochs 1
python code/nlg/run_qwen_nlg_generation.py --task e2e --model_name gpt2-medium --method lora --prompt_variant strict --output_dir results/nlg/example_gpt2_e2e_lora --max_train_examples 128 --max_eval_examples 64 --epochs 1
```

## 6. Results / Insights

LoRA matched FT closely on several RoBERTa GLUE and ViT vision settings while using under 1M trainable parameters. Our GPT-2 E2E reproduction had lower absolute BLEU/ROUGE than the original paper, likely due to practical differences in data scale, training schedule, prompting, decoding, and post-processing.

In extensions, LoRA transferred well to vision, audio results were task-dependent, and LoRA often had lower ECE than FT on GLUE reliability experiments. Detailed artifacts are stored under `results/`, and the polished comparison tables are in `report/4782_Final_Project_Report.pdf`.

## 7. Conclusion

LoRA is highly parameter-efficient, but it is not simply "full fine-tuning with fewer parameters." Its strengths depend on task, modality, metric, and adaptation difficulty.

The main lesson from the re-implementation is that matching the original paper requires careful control of preprocessing, training budget, decoding, and evaluation details; small practical differences can noticeably change NLG scores.

## 8. References

Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., and Chen, W. (2022). **LoRA: Low-rank adaptation of large language models.** ICLR.

Liu, Y. et al. (2019). **RoBERTa: A robustly optimized BERT pretraining approach.** arXiv:1907.11692.

Radford, A. et al. (2019). **Language models are unsupervised multitask learners.** OpenAI technical report.

Wolf, T. et al. (2020). **Transformers: State-of-the-art natural language processing.** EMNLP System Demonstrations.

Guo, C., Pleiss, G., Sun, Y., and Weinberger, K. Q. (2017). **On calibration of modern neural networks.** ICML.

## 9. Acknowledgements

This project was completed as a CS 4782 final project at Cornell University. We thank the course staff for the project structure and feedback, and the Hugging Face ecosystem for model and dataset tooling used throughout the experiments.
