# Audio LoRA Result Table

| experiment | task_name | method | model_name | accuracy | eval_loss | trainable_parameters | # Trainable Parameters | trainable_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| minds14_en_ft | minds14_en | ft | facebook/wav2vec2-base | 0.097345 | 2.626224 | 94572174 | 94.57M | 1.000000 |
| minds14_en_lora | minds14_en | lora | facebook/wav2vec2-base | 0.141593 | 2.575032 | 347918 | 0.35M | 0.003673 |
| speech_commands_ft | speech_commands | ft | facebook/wav2vec2-base | 0.977259 | 0.090014 | 94577828 | 94.58M | 1.000000 |
| speech_commands_lora | speech_commands | lora | facebook/wav2vec2-base | 0.964336 | 0.125546 | 353572 | 0.35M | 0.003733 |
| superb_er_ft | superb_er | ft | facebook/wav2vec2-base | 0.641418 | 0.942119 | 94569604 | 94.57M | 1.000000 |
| superb_er_lora | superb_er | lora | facebook/wav2vec2-base | 0.547945 | 1.065110 | 345348 | 0.35M | 0.003646 |
