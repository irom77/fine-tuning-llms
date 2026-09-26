# Task 4 Results: Train with LoRA

## Purpose

Task 4 fine-tunes the TacoBot model using the LoRA configuration created in
Task 3. The base model remains frozen while only the LoRA adapter parameters
are updated.

## Execution

The task was run from the repository root with the CPU virtual environment:

```bash
cd /home/irom/fine-tuning-llms
source .venv/bin/activate
python task_4_train_lora.py
```

Training completed successfully on the laptop CPU. A GPU was not required.
The complete 50-step run took approximately 3.5 minutes.

## Training configuration

| Setting | Value |
|---|---:|
| Base model | `HuggingFaceTB/SmolLM2-135M-Instruct` |
| Training examples | `22` |
| LoRA rank | `8` |
| LoRA alpha | `16` |
| Trainable parameters | `460,800` |
| Trainable percentage | `0.34%` |
| Maximum steps | `50` |
| Batch size | `1` |
| Gradient accumulation steps | `4` |
| Learning rate | `0.0002` |
| Maximum sequence length | `256` tokens |
| Device | CPU |

The LoRA configuration was loaded from `markers/lora_config.json`.

## Observed results

The run reported:

```text
Training steps: 50
Final loss: 2.4245
Training runtime: 211 seconds
Training samples per second: 0.948
Training steps per second: 0.237
```

The logged loss decreased during training. Logged checkpoints included:

| Step | Logged loss |
|---:|---:|
| 5 | 2.814 |
| 10 | 2.680 |
| 20 | 2.525 |
| 30 | 2.324 |
| 40 | 2.218 |
| 50 | 2.182 |

The final reported training loss is the aggregate value `2.4245`, while the
step-50 log is the loss for the final logging interval.

### STEP 5 console snippet

The central training step produced output similar to:

```text
[STEP 5] Training with LoRA...
-----------------------------------------------------------------
  Watch the loss decrease as the model learns!

  0%|                                                   | 0/50 [00:00<?, ?it/s]
 10%|████▎                                      | 5/50 [00:27<03:58, 5.30s/it]
{'loss': '2.814', 'grad_norm': '0.6117', 'learning_rate': '0.000184', 'epoch': '0.9091'}
 50%|█████████████████████                     | 25/50 [01:48<01:45, 4.23s/it]
{'loss': '2.379', 'grad_norm': '0.6966', 'learning_rate': '0.000104', 'epoch': '4.182'}
 80%|█████████████████████████████████▌        | 40/50 [02:52<00:44, 4.49s/it]
{'loss': '2.218', 'grad_norm': '0.7925', 'learning_rate': '0.000044', 'epoch': '6.727'}
100%|██████████████████████████████████████████| 50/50 [03:30<00:00, 3.80s/it]
{'loss': '2.182', 'grad_norm': '0.8398', 'learning_rate': '0.000004', 'epoch': '8.364'}

{'train_runtime': '211', 'train_samples_per_second': '0.948',
 'train_steps_per_second': '0.237', 'train_loss': '2.425',
 'epoch': '8.364'}
```

Progress-bar timing and wording can vary slightly between runs, especially on
CPU. The important indicators are that all 50 steps complete and the loss
values are reported.

## Output artifacts

Successful execution creates:

```text
lora_output/
lora_adapter/
markers/task4_complete.txt
```

The trained adapter directory is approximately `5.14 MB` and contains the
LoRA weights, adapter configuration, and tokenizer files. The marker contains
the completion status and final aggregate loss:

```text
TRAIN_LORA_COMPLETE
loss=2.4245
```

## Environment-path fix

The original script referenced `/root/code/data/training_data.jsonl`,
`/root/lora_output`, `/root/lora_adapter`, and `/root/markers`. These paths
were changed to repository-relative locations so the task works in this
checkout without special permissions.

## Reproducing the task

Run:

```bash
cd /home/irom/fine-tuning-llms
.venv/bin/python task_4_train_lora.py
```

CPU training is functional but slower than CUDA training. Re-running the task
will overwrite the adapter output with the new run's weights.

## Key takeaway

The run successfully trained a small adapter instead of updating all
134-million-plus base-model parameters. This demonstrates why LoRA makes
fine-tuning feasible on consumer hardware with limited compute resources.
