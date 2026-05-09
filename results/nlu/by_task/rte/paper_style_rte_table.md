# Four-Task Paper-Style Result Table

`Avg4` is the average over the selected four tasks, not the original paper's 8-task GLUE average.

| Model & Method | # Trainable Parameters | rte | Avg4 |
| --- | --- | --- | --- |
| RoBbase (Adapter, size=16) | 0.90M | 0.7004 | 0.7004 |
| RoBbase (Adapter, size=48) | 1.49M | 0.7184 | 0.7184 |
| RoBbase (BitFit) | 0.69M | 0.6101 | 0.6101 |
| RoBbase (FT) | 124.65M | 0.7762 | 0.7762 |
| RoBbase (LoRA r=4) | 0.74M | 0.6931 | 0.6931 |
