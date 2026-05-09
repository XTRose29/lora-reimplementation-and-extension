# Task-Probed LoRA Idea

This folder documents a possible small extension beyond our standard LoRA reimplementation. The current idea is **Task-Probed LoRA**: before training LoRA, run a cheap probe pass to decide where LoRA adapters should be placed.

## Code in This Folder

The runnable NLU pipeline is documented in `README_NLU.md`.

Main files:

| File | Role |
|---|---|
| `probe_lora.py` | scores candidate modules and injects LoRA only into selected modules |
| `train_task_probed_lora_nlu.py` | runs probe + selective LoRA training on GLUE |
| `evaluate_task_probed_lora_nlu.py` | reloads the selected LoRA checkpoint and evaluates it |
| `plot_probe_results.py` | writes summary tables and simple figures |
| `run_one_task_probe.ps1` | runs all probe strategies on one GLUE task |
| `run_four_task_probe.ps1` | runs SST-2, MRPC, RTE, and CoLA |

Quick command:

```powershell
.\newimpl\run_one_task_probe.ps1 -TaskName sst2 -ModelName distilroberta-base -TopK 4 -Quick
```

## Core Question

Standard LoRA usually adds adapters to every chosen projection, such as all RoBERTa attention `query` and `value` layers. That assumes every layer is equally useful for adaptation.

Our question is simpler:

> Can we use a small task-specific probe to find which modules deserve LoRA adapters?

## Storyline

Gradient tells us **where the task wants change**.

Activation tells us **where the model is actually working**.

Task-Probed LoRA places adapters where both signals agree.

In other words, a good LoRA location should not only have a large gradient, and should not only have a large activation. It should be both active in the forward pass and sensitive to the supervised task loss.

## Probe Signals

For each candidate module, such as a RoBERTa attention `query` or `value` projection:

```text
gradient_energy = ||dL / dW||
activation_energy = ||module_output||
```

Then combine them:

```text
score = normalized(gradient_energy) * normalized(activation_energy)
```

or a slightly smoother version:

```text
score = sqrt(normalized(gradient_energy) * normalized(activation_energy))
```

High score means:

- the module is used by the model on this task's inputs;
- the task loss wants this module to change;
- this module is a reasonable place to insert LoRA.

## Minimal Experiment

Candidate modules:

```text
roberta.encoder.layer.*.attention.self.query
roberta.encoder.layer.*.attention.self.value
```

For RoBERTa-base, this gives 24 candidates: 12 layers times query/value.

Methods to compare:

| Method | Meaning |
|---|---|
| Full LoRA | Add LoRA to all 24 query/value modules |
| Random top-k | Randomly choose k modules |
| Gradient top-k | Choose k modules by gradient energy |
| Activation top-k | Choose k modules by activation energy |
| ActGrad top-k | Choose k modules by combined activation + gradient score |

Suggested tasks:

```text
SST-2, MRPC, RTE, CoLA
```

Suggested probe size:

```text
512 or 1024 training examples
```

Suggested k:

```text
4, 8, 12
```

## Expected Takeaway

If ActGrad top-k is close to Full LoRA while using fewer adapter locations, the result supports the idea that many LoRA locations are redundant.

If ActGrad beats Random top-k, the probe signal is doing something useful.

If ActGrad beats Gradient-only or Activation-only, the combined signal is more reliable than either signal alone.

## Scope

This is not a full new LoRA algorithm yet. It is a lightweight extension for our course project:

> Standard LoRA reproduction + a task-sensitive adapter placement study.

For a larger research version, this could become a budgeted LoRA allocation method that assigns different ranks to different modules based on probe scores.
