# Four-Task Paper-Style Result Table

`Avg4` is the average over the selected four tasks, not the original paper's 8-task GLUE average.

| Model & Method | # Trainable Parameters | cola | Avg4 |
| --- | --- | --- | --- |
| RoBbase (Adapter, size=16) | 0.90M | 0.5677 | 0.5677 |
| RoBbase (Adapter, size=48) | 1.49M | 0.6032 | 0.6032 |
| RoBbase (BitFit) | 0.69M | 0.5100 | 0.5100 |
| RoBbase (FT) | 124.65M | 0.5932 | 0.5932 |
| RoBbase (LoRA r=4) | 0.74M | 0.5933 | 0.5933 |
