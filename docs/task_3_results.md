# Task 3 Results: Configure and Apply LoRA

## Purpose

Task 3 configures Low-Rank Adaptation (LoRA) for the TacoBot language model.
LoRA freezes the base model and adds small trainable adapter matrices to
selected attention projections. This reduces the number of parameters that
need to be updated during fine-tuning.

## Execution

The task was run from the repository root with the CPU virtual environment:

```bash
cd /home/irom/fine-tuning-llms
source .venv/bin/activate
python task_3_configure_lora.py
```

The model loaded and LoRA was applied successfully on the laptop CPU. A GPU
is not required for this configuration step.

## LoRA configuration

The script uses:

| Setting | Value |
|---|---:|
| Base model | `HuggingFaceTB/SmolLM2-135M-Instruct` |
| LoRA rank (`r`) | `8` |
| LoRA alpha | `16` |
| LoRA dropout | `0.05` |
| Target modules | `q_proj`, `v_proj` |
| Bias | `none` |
| Task type | Causal language modeling |

The alpha is twice the rank, giving a common starting configuration that
balances adapter capacity and efficiency.

## Observed results

The verified run reported:

```text
Total parameters: 134,515,008
Frozen (base model): 134,515,008
Trainable (adapters): 460,800
Percentage trainable: 0.34%
```

The model contained 120 LoRA parameter entries across the targeted
projections. The script estimated approximately 1,539 MB for full FP32
fine-tuning and approximately 5 MB for the adapter parameters using its
simplified memory calculation.

The displayed memory reduction rounds to `100%`; the value is an estimate of
parameter, gradient, and optimizer memory and is not a complete measurement
of runtime memory usage.

## Output artifacts

Successful execution creates:

```text
markers/lora_config.json
markers/task3_complete.txt
```

`markers/lora_config.json` records the rank, alpha, target modules, trainable
parameter count, and total parameter count for use by the next task.

`markers/task3_complete.txt` contains:

```text
CONFIGURE_LORA_COMPLETE
```

## Environment-path fix

The original lab script wrote its outputs to `/root/markers`, which is not a
writable or appropriate location for this checkout. The script now resolves
the repository root from its own file location and writes both artifacts to
the local `markers/` directory.

## Reproducing the task

Run:

```bash
cd /home/irom/fine-tuning-llms
.venv/bin/python task_3_configure_lora.py
```

The model is cached after its first download, so later runs should avoid
downloading it again.

## Key takeaway

LoRA allows the lab to adapt the model while training only `0.34%` of its
parameters. This makes the configuration suitable for experimentation on
consumer hardware, although the later training task will still require
additional CPU time and memory.

