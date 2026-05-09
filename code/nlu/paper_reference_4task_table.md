# Paper Reference Table, Four Selected Tasks

This is the subset of the LoRA paper table for the four tasks we plan to run locally.

`Avg4` is the average over only these four tasks. It is not the paper's original 8-task GLUE average.

| Model & Method | # Trainable Parameters | SST-2 | MRPC | CoLA | RTE | Avg4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RoBbase (FT)* | 125.0M | 94.8 | 90.2 | 63.6 | 78.7 | 81.8 |
| RoBbase (BitFit)* | 0.1M | 93.7 | 92.7 | 62.0 | 81.5 | 82.5 |
| RoBbase (AdptD)* | 0.3M | 94.2 | 88.5 | 60.8 | 71.5 | 78.8 |
| RoBbase (AdptD)* | 0.9M | 94.7 | 88.4 | 62.6 | 75.9 | 80.4 |
| RoBbase (LoRA) | 0.3M | 95.1 | 89.7 | 63.4 | 86.6 | 83.7 |

Our generated table will be written to:

```text
my_NLU/results/four_task_table/paper_style_4task_table.md
```

