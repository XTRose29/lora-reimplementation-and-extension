# LoRA Re-implementation and Extension

**Authors:** Rose Tianruo Xu and Weijie Zhou  
**Contribution statement:** Both authors contributed equally to this project.

## 1. Introduction

This repository contains a CS 4782 final project re-implementing and extending **LoRA: Low-Rank Adaptation of Large Language Models**.

LoRA freezes pretrained weights and trains a low-rank update, `Delta W = (alpha / r) BA`, reducing trainable parameters while aiming to preserve full fine-tuning performance.

## 2. Chosen Result

We reproduce the LoRA paper's core NLU claim: LoRA on RoBERTa-base can match or approach full fine-tuning while training orders of magnitude fewer parameters.

Our main reproduction corresponds to the original paper's full fine-tuning vs adapter vs LoRA comparison on GLUE-style NLU tasks. We then extend the comparison to vision, audio, reliability/calibration, and structured NLG generation.

## 3. GitHub Contents

- `code/`: NLU, vision, audio, reliability, task-probed LoRA, and NLG experiment code.
- `data/`: Lightweight dataset previews plus instructions for obtaining full datasets.
- `results/`: Metrics, logs, predictions, figures, and report-level summary tables.
- `poster/`: Final in-class poster PDF.
- `report/`: Final two-page project report PDF.
- `LICENSE`: MIT license for this project repository.

## 4. Re-implementation Details

NLU experiments compare full fine-tuning, adapters, BitFit, and LoRA on SST-2, MRPC, RTE, and CoLA using RoBERTa-base-style sequence classification.

Extensions evaluate ViT on vision tasks, wav2vec2 on audio tasks, GLUE reliability/calibration metrics, and Qwen2.5-0.5B-Instruct NLG generation on E2E/WebNLG with DART smoke tests.

## 5. Reproduction Steps

Set up the environment from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r code/requirements.txt
export PYTHONPATH="$PWD/code:$PYTHONPATH"
```

Example runs:

```bash
python code/reimpl/train_my_lora_nlu.py --task_name sst2 --method lora --output_dir results/nlu/example_sst2_lora
python code/vision/train_my_lora_vision.py --task_name cifar10 --method lora --output_dir results/vision/example_cifar10_lora
python code/audio/train_my_lora_audio.py --task_name speech_commands --method lora --output_dir results/audio/example_speech_commands_lora
python code/reliability/run_cola_reliability.py --task_name cola --method lora --output_dir results/reliability/example_cola_lora
bash code/nlg/run_nlg_smoke_matrix.sh
```

A CUDA-capable GPU is recommended for full reproduction; CPU runs are suitable only for small smoke tests.

## 6. Results/Insights

On NLU, LoRA `r=4` trains 0.74M parameters versus 124.65M for full fine-tuning, while matching FT on SST-2 and CoLA and trailing more on RTE.

Vision LoRA nearly matches FT across CIFAR-10, Beans, and Oxford-IIIT Pet, while audio LoRA is cheaper but weaker than FT across the wav2vec2 tasks.

Reliability experiments show LoRA often lowers ECE and improves selective accuracy, suggesting a calibration benefit even when primary task score drops slightly.

NLG experiments show a scale-dependent pattern: LoRA is more format-stable in small smoke tests, but full fine-tuning becomes stronger on E2E and WebNLG when trained on 5,000 examples.

## 7. Conclusion

LoRA is often close to full fine-tuning at a tiny fraction of the trainable parameter count, but it is not simply full fine-tuning made cheaper.

The extensions show that low-rank adaptation works especially well for many NLU and vision settings, while audio and larger-scale generation expose cases where full fine-tuning's extra flexibility matters.

## 8. References

- Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. **LoRA: Low-Rank Adaptation of Large Language Models**. ICLR 2022.
- Alex Wang et al. **GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding**. ICLR 2019.
- Alexey Dosovitskiy et al. **An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale**. ICLR 2021.
- Alexei Baevski et al. **wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations**. NeurIPS 2020.
- Jekaterina Novikova et al. **The E2E Dataset: New Challenges for End-to-End Generation**. SIGDIAL 2017.
- Claire Gardent et al. **Creating Training Corpora for NLG Micro-Planners**. ACL 2017.

## 9. Acknowledgements

This work was completed as a final project for CS 4782.

We thank the course staff and project collaborators for feedback, guidance, and evaluation.

