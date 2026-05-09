| task_name | method | model_name | epochs | batch_size | max_train_samples | max_eval_samples | lora_r | lora_alpha | lora_dropout | lora_placement | attention_targets | abstention_threshold | calibration_bins | seed | output_dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cola | ft | Qwen/Qwen2.5-0.5B-Instruct | 5 | 16 |  |  |  |  |  |  |  | 0.80 | 15 | 42 | cola_reliability/results/exp1_reasoning_0p5b_ft_vs_lora/cola_ft |
| cola | lora | Qwen/Qwen2.5-0.5B-Instruct | 5 | 16 |  |  | 4 | 32 | 0.1 | attention | auto | 0.80 | 15 | 42 | cola_reliability/results/exp1_reasoning_0p5b_ft_vs_lora/cola_lora_r4_attention |
| mrpc | ft | Qwen/Qwen2.5-0.5B-Instruct | 5 | 16 |  |  |  |  |  |  |  | 0.80 | 15 | 42 | cola_reliability/results/exp1_reasoning_0p5b_ft_vs_lora/mrpc_ft |
| mrpc | lora | Qwen/Qwen2.5-0.5B-Instruct | 5 | 16 |  |  | 4 | 32 | 0.1 | attention | auto | 0.80 | 15 | 42 | cola_reliability/results/exp1_reasoning_0p5b_ft_vs_lora/mrpc_lora_r4_attention |
| rte | ft | Qwen/Qwen2.5-0.5B-Instruct | 5 | 16 |  |  |  |  |  |  |  | 0.80 | 15 | 42 | cola_reliability/results/exp1_reasoning_0p5b_ft_vs_lora/rte_ft |
| rte | lora | Qwen/Qwen2.5-0.5B-Instruct | 5 | 16 |  |  | 4 | 32 | 0.1 | attention | auto | 0.80 | 15 | 42 | cola_reliability/results/exp1_reasoning_0p5b_ft_vs_lora/rte_lora_r4_attention |
| sst2 | ft | Qwen/Qwen2.5-0.5B-Instruct | 5 | 16 |  |  |  |  |  |  |  | 0.80 | 15 | 42 | cola_reliability/results/exp1_reasoning_0p5b_ft_vs_lora/sst2_ft |
| sst2 | lora | Qwen/Qwen2.5-0.5B-Instruct | 5 | 16 |  |  | 4 | 32 | 0.1 | attention | auto | 0.80 | 15 | 42 | cola_reliability/results/exp1_reasoning_0p5b_ft_vs_lora/sst2_lora_r4_attention |
