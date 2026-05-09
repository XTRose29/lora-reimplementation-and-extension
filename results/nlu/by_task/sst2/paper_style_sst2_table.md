# Four-Task Paper-Style Result Table

`Avg4` is the average over the selected four tasks, not the original paper's 8-task GLUE average.

| Model & Method | # Trainable Parameters | sst2 | Avg4 |
| --- | --- | --- | --- |
| RoBbase (Adapter, size=16) | 0.90M | 0.9358 | 0.9358 |
| RoBbase (Adapter, size=48) | 1.49M | 0.9369 | 0.9369 |
| RoBbase (BitFit) | 0.69M | 0.9358 | 0.9358 |
| RoBbase (FT) | 124.65M | 0.9404 | 0.9404 |
| RoBbase (LoRA r=4) | 0.74M | 0.9392 | 0.9392 |
