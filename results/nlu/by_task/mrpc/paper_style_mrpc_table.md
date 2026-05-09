# Four-Task Paper-Style Result Table

`Avg4` is the average over the selected four tasks, not the original paper's 8-task GLUE average.

| Model & Method | # Trainable Parameters | mrpc | Avg4 |
| --- | --- | --- | --- |
| RoBbase (Adapter, size=16) | 0.90M | 0.9101 | 0.9101 |
| RoBbase (Adapter, size=48) | 1.49M | 0.9137 | 0.9137 |
| RoBbase (BitFit) | 0.69M | 0.8548 | 0.8548 |
| RoBbase (FT) | 124.65M | 0.9236 | 0.9236 |
| RoBbase (LoRA r=4) | 0.74M | 0.9037 | 0.9037 |
