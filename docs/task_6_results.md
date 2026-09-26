# Task 6 Results: Create DPO Preference Data

## Purpose

Task 6 prepares preference pairs for Direct Preference Optimization (DPO).
Each record contains a customer prompt, a helpful `chosen` response, and an
intentionally poor `rejected` response.

## Execution

The task was run twice from the repository root with the CPU virtual
environment:

```bash
cd /home/irom/fine-tuning-llms
source .venv/bin/activate
python task_6_create_dpo_data.py
python task_6_create_dpo_data.py
```

## Results

| Run | Records loaded | Result | Records after run | Backup |
|---|---:|---|---:|---|
| First | 10 | Added one new preference pair | 11 | Created |
| Second | 11 | Detected existing pair; no duplicate added | 11 | Not needed |

The generated pair describes a customer waiting 20 minutes for food. The
chosen response apologizes, offers to check the order, and provides a free
drink. The rejected response is intentionally dismissive.

## Backup

Before the first modification, the original dataset was preserved at:

```text
data/backups/task6/preference_pairs_20260926T144001Z.json
```

Future modifications create additional timestamped backups in the same
directory. Existing identical pairs are skipped, so rerunning the task does
not grow the dataset with duplicates.

## Validation

The script verified that:

- The preference file contains a JSON list.
- Every pair has non-empty `prompt`, `chosen`, and `rejected` fields.
- The chosen and rejected responses differ.
- The generated pair occurs exactly once.
- The completion marker was written successfully.

## Output artifact

Successful execution writes:

```text
markers/task6_complete.txt
```

The marker contains:

```text
CREATE_DPO_DATA_COMPLETE
```

