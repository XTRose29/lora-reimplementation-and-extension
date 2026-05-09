# Vision LoRA Result Table

| experiment | task_name | method | accuracy | # Trainable Parameters | trainable_parameters | trainable_ratio | eval_loss | model_name |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| beans_ft | beans | ft | 0.9925 | 85.80M | 85800963 | 1.000000 | 0.0933 | google/vit-base-patch16-224-in21k |
| beans_lora | beans | lora | 0.9925 | 0.15M | 149763 | 0.001742 | 0.0338 | google/vit-base-patch16-224-in21k |
