# Task-Probed LoRA 想法说明

这个文件夹记录一个可能的小创新方向。当前想法叫 **Task-Probed LoRA**：在正式训练 LoRA 之前，先用很少的数据做一次便宜的 probe，判断 LoRA 应该插在哪里。

## 这个文件夹里的代码

可运行的 NLU pipeline 说明在 `README_NLU_CN.md`。

主要文件：

| 文件 | 作用 |
|---|---|
| `probe_lora.py` | 给候选模块打分，并且只在选中的模块里插 LoRA |
| `train_task_probed_lora_nlu.py` | 跑 probe + selective LoRA 的 GLUE 训练 |
| `evaluate_task_probed_lora_nlu.py` | 重新加载选中位置的 LoRA checkpoint 并评估 |
| `plot_probe_results.py` | 汇总结果，画简单图 |
| `run_one_task_probe.ps1` | 在一个 GLUE task 上跑所有 probe 策略 |
| `run_four_task_probe.ps1` | 跑 SST-2、MRPC、RTE、CoLA 四个任务 |

快速测试命令：

```powershell
.\newimpl\run_one_task_probe.ps1 -TaskName sst2 -ModelName distilroberta-base -TopK 4 -Quick
```

## 核心问题

标准 LoRA 通常会把 adapter 加到所有指定位置，比如 RoBERTa 每一层 attention 的 `query` 和 `value`。

这等于默认假设：

> 每个 query/value 位置都同样值得被改。

我们的想法是问一个更细的问题：

> 能不能先用一个小 probe，找出这个任务真正需要 LoRA 的位置？

## 故事线

Gradient 告诉我们：**任务想改哪里**。

Activation 告诉我们：**模型正在用哪里**。

Task-Probed LoRA 的直觉是：

> LoRA 应该放在两者都认可的位置。

也就是说，一个位置只靠 gradient 大还不够，因为它可能只是某个 batch 的噪声；只靠 activation 大也不够，因为它可能很活跃但不需要被改。真正值得放 LoRA 的地方应该是：

```text
模型确实在用它 + 任务 loss 确实想改它
```

## Probe 信号

对每个候选模块，比如 RoBERTa attention 里的某个 `query` 或 `value`：

```text
gradient_energy = ||dL / dW||
activation_energy = ||module_output||
```

然后组合成一个分数：

```text
score = normalized(gradient_energy) * normalized(activation_energy)
```

或者更平滑一点：

```text
score = sqrt(normalized(gradient_energy) * normalized(activation_energy))
```

分数高表示：

- 这个模块在当前任务输入里真的参与了计算；
- 任务 loss 真的希望它变化；
- 它是一个合理的 LoRA 插入点。

## 一个简单比喻

只看 gradient：

> 老师批作业时哪里红圈最多。

只看 activation：

> 课堂上谁说话最多。

ActGrad score：

> 谁既经常参与讨论，又确实在关键问题上需要被纠正。

所以它比单独看 gradient 或 activation 更稳。

## 最小实验设计

候选模块：

```text
roberta.encoder.layer.*.attention.self.query
roberta.encoder.layer.*.attention.self.value
```

RoBERTa-base 有 12 层，所以一共有：

```text
12 layers x query/value = 24 个候选位置
```

比较方法：

| 方法 | 含义 |
|---|---|
| Full LoRA | 所有 24 个 query/value 都加 LoRA |
| Random top-k | 随机选 k 个位置加 LoRA |
| Gradient top-k | 只按 gradient energy 选 k 个 |
| Activation top-k | 只按 activation energy 选 k 个 |
| ActGrad top-k | 按 activation + gradient 组合分数选 k 个 |

建议任务：

```text
SST-2, MRPC, RTE, CoLA
```

建议 probe 数据量：

```text
512 或 1024 条训练样本
```

建议 k：

```text
4, 8, 12
```

## 期待看到什么

如果 ActGrad top-k 接近 Full LoRA，但用了更少的 LoRA 位置：

> 说明很多 LoRA 插入点可能是冗余的。

如果 ActGrad 比 Random top-k 好：

> 说明 probe 不是随机碰运气，而是真的找到了一些有用位置。

如果 ActGrad 比 Gradient-only 或 Activation-only 好：

> 说明“任务想改哪里”和“模型正在用哪里”结合起来，比单一信号更可靠。

## 项目定位

这个目前不是完整的新 LoRA 算法，而是课程项目里的一个轻量 extension：

> 先复刻标准 LoRA，再研究一个 task-sensitive 的 LoRA 插入位置选择方法。

如果以后想扩展成更大的研究，可以继续做成：

> 根据 probe score 给不同模块分配不同 rank 的 budgeted LoRA allocation 方法。
