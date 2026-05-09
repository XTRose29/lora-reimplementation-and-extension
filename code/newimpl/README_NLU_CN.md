# 新 NLU Pipeline：Task-Probed LoRA

这个 pipeline 用来测试 `newimpl/` 里的小创新：正式训练 LoRA 之前，先用少量数据做 probe，判断 LoRA 应该插在 RoBERTa attention 的哪些位置。

## 会跑什么

每个 GLUE task 会比较 5 种策略：

| 策略 | 含义 |
|---|---|
| `full` | 所有 attention query/value 都加 LoRA |
| `random` | 随机选 `top_k` 个 query/value 加 LoRA |
| `gradient` | 按 gradient energy 选 `top_k` |
| `activation` | 按 activation energy 选 `top_k` |
| `actgrad` | 按 activation + gradient 组合分数选 `top_k` |

默认任务：

```text
sst2, mrpc, rte, cola
```

默认候选模块：

```text
roberta.encoder.layer.*.attention.self.query
roberta.encoder.layer.*.attention.self.value
```

## 安装

在 repo 根目录运行：

```powershell
pip install -r reimpl\requirements.txt
```

## 快速测试

先跑一个很小的 SST-2，确认代码能跑通。这个不是最终实验结果。

```powershell
.\newimpl\run_one_task_probe.ps1 -TaskName sst2 -ModelName roberta-base -TopK 8 -Quick
```

如果只是 CPU/debug，可以用小一点的模型：

```powershell
.\newimpl\run_one_task_probe.ps1 -TaskName sst2 -ModelName distilroberta-base -TopK 4 -Quick
```

## 正式运行

跑单个任务：

```powershell
.\newimpl\run_one_task_probe.ps1 -TaskName rte -ModelName roberta-base -TopK 8
```

跑四个任务：

```powershell
.\newimpl\run_four_task_probe.ps1 -ModelName roberta-base -TopK 8
```

## 五个主要命令

正式跑实验时，优先看这五个命令。

四个 task 合并跑：

```powershell
.\newimpl\run_four_task_probe.ps1 -ModelName roberta-base -TopK 8
```

四个 task 分开跑：

```powershell
.\newimpl\run_one_task_probe.ps1 -TaskName sst2 -ModelName roberta-base -TopK 8
```

```powershell
.\newimpl\run_one_task_probe.ps1 -TaskName mrpc -ModelName roberta-base -TopK 8
```

```powershell
.\newimpl\run_one_task_probe.ps1 -TaskName rte -ModelName roberta-base -TopK 8
```

```powershell
.\newimpl\run_one_task_probe.ps1 -TaskName cola -ModelName roberta-base -TopK 8
```

后台跑四个任务：

```powershell
$repo = (Get-Location).Path
Start-Job -ArgumentList $repo -ScriptBlock { param($repo) Set-Location $repo; .\newimpl\run_four_task_probe.ps1 -ModelName roberta-base -TopK 8 }
```

## 手动跑单个策略

如果只想跑 ActGrad 一个策略：

```powershell
python newimpl\train_task_probed_lora_nlu.py `
  --task_name sst2 `
  --model_name roberta-base `
  --output_dir newimpl\results\nlu_probe\sst2\sst2_actgrad_top8 `
  --probe_strategy actgrad `
  --top_k 8 `
  --probe_samples 1024 `
  --candidate_scope attention `
  --target_modules query,value `
  --lora_r 4 `
  --lora_alpha 32 `
  --lora_dropout 0.1 `
  --epochs 3 `
  --batch_size 16 `
  --learning_rate 0.0002

python newimpl\evaluate_task_probed_lora_nlu.py `
  --checkpoint_dir newimpl\results\nlu_probe\sst2\sst2_actgrad_top8\checkpoint `
  --output_dir newimpl\results\nlu_probe\sst2\sst2_actgrad_top8
```

## 输出文件

每个实验文件夹会有：

| 文件 | 作用 |
|---|---|
| `probe_scores.json` | 每个候选模块的 activation、gradient、ActGrad 分数 |
| `probe_selection.json` | 最后被选中插 LoRA 的模块 |
| `train_config.json` | 完整训练和 probe 配置 |
| `parameter_count.json` | 可训练参数量 |
| `metrics.json` / `eval.txt` | 验证集指标 |
| `checkpoint/` | LoRA 和分类头 checkpoint |

汇总结果在：

```text
newimpl/results/nlu_probe/probe_summary_table.csv
newimpl/results/nlu_probe/figures/
```

## 最适合写进报告的比较

建议主表放：

```text
Full LoRA
Random top-8
Gradient top-8
Activation top-8
ActGrad top-8
```

如果 `ActGrad top-8` 接近 `Full LoRA`，同时好过 `Random top-8`，说明这个 probe 确实找到了更有用的 LoRA 插入位置。
