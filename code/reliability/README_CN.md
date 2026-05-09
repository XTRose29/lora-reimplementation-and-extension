# CoLA LoRA Reliability Experiments

这个文件夹用于研究：

> When does LoRA become overconfident, and when does it remain reliable?

核心设置：

- base model: `roberta-base`
- task: GLUE CoLA
- baseline: full fine-tuning, `--method ft`
- LoRA: 使用本项目 `reimpl/my_lora.py` 里的 `MyLoRALinear`
- main metrics: accuracy / MCC, ECE, calibration_accuracy, NLL, Brier, abstention_rate, selective_accuracy

## 指标怎么算

### Calibration

脚本使用 max softmax probability 作为模型 confidence：

```text
confidence_i = max_k softmax(logits_i)_k
prediction_i = argmax_k softmax(logits_i)_k
correct_i = 1[prediction_i = label_i]
```

ECE, expected calibration error：

```text
ECE = sum_b (|B_b| / N) * | accuracy(B_b) - confidence(B_b) |
```

其中 `B_b` 是按 confidence 分出来的 bin，默认 15 个 bin。ECE 越低越好。

为了对应你说的“calibration 准确度”，脚本额外报告：

```text
calibration_accuracy = 1 - ECE
```

所以 calibration_accuracy 越高越好。但论文里更常见、也更严谨的写法还是直接报告 ECE / NLL / Brier。

### Abstention

脚本把低于阈值的样本视为模型选择“不回答”：

```text
attempt_i = confidence_i >= threshold
abstention_rate = mean(confidence_i < threshold)
coverage = 1 - abstention_rate
selective_accuracy = accuracy on attempted examples
```

默认 `threshold = 0.80`。如果 LoRA 更 overconfident，常见现象会是：

- ECE 更高
- mean_confidence 更高但 accuracy/MCC 没同步提高
- abstention_rate 更低，也就是更“敢答”
- selective_accuracy 不一定更好

## 主要实验问题

### 1. FT vs LoRA

先比较 full fine-tuning 和标准 LoRA：

```powershell
powershell -ExecutionPolicy Bypass -File .\cola_reliability\run_quick_smoke.ps1
```

完整一点：

```powershell
python .\cola_reliability\run_cola_reliability.py `
  --method ft `
  --model_name roberta-base `
  --output_dir cola_reliability\results\ft_full `
  --epochs 5 `
  --batch_size 16

python .\cola_reliability\run_cola_reliability.py `
  --method lora `
  --model_name roberta-base `
  --output_dir cola_reliability\results\lora_r4_attention_full `
  --lora_r 4 `
  --lora_placement attention `
  --epochs 5 `
  --batch_size 16
```

实验假设：

> 在 matched 或接近 matched task performance 下，LoRA 的 ECE 会高于 FT，`calibration_accuracy = 1 - ECE` 会低于 FT，并且 abstention_rate 可能更低。

## 对比实验矩阵

一键运行矩阵：

```powershell
powershell -ExecutionPolicy Bypass -File .\cola_reliability\run_full_matrix.ps1
```

快速矩阵：

```powershell
powershell -ExecutionPolicy Bypass -File .\cola_reliability\run_full_matrix.ps1 -Quick
```

矩阵覆盖：

- training 数据量：`512`, `2000`, `8551`
- LoRA rank：`1`, `4`, `8`, `16`
- LoRA placement：`attention`, `mlp`, `attention_mlp`
- layer groups：`0-3`, `4-7`, `8-11`
- ID: CoLA validation
- OOD: 默认用 SST-2 validation 做 confidence-only OOD 检查

注意：默认 OOD 是 confidence-only，因为 SST-2 的 label 是 sentiment，不是 grammatical acceptability，不能直接拿来算 CoLA accuracy/ECE。它可以回答“模型在非 CoLA 输入上是否仍然很自信、是否更少 abstain”。如果你有兼容 CoLA 标签的 OOD acceptability 数据，可以用 `--ood_has_compatible_labels` 开启 OOD accuracy/ECE。

## 输出文件

每个 run 会写：

- `train_config.json`
- `parameter_count.json`
- `train_log.jsonl`
- `id_metrics.json`
- `ood_metrics.json`
- `id_predictions.jsonl`
- `ood_predictions.jsonl`
- `metrics.json`

汇总：

```powershell
python .\cola_reliability\summarize_results.py --results_root cola_reliability\results
```

输出：

- `cola_reliability/results/summary_table.csv`
- `cola_reliability/results/summary_table.md`

## 和参考论文的关系

这个实验不是再证明“LoRA 的 ECE 高/低”这个单点结论，而是把问题条件化：

- 小数据时是否更容易 overconfident？
- rank 越大是否越容易破坏 calibration？
- attention-only 和 MLP-only 哪个更影响 confidence？
- 哪些层的 LoRA 更新更容易导致 reliability failure？
- 在 OOD 输入上 LoRA 是否更 answer-first，也就是 abstention_rate 更低？

这对应一个更清楚的分析型题目：

> Does LoRA encourage answer-first behavior? A study of calibration and abstention under matched task performance.
