# Task 2 Results: Prepare Training Data

## Purpose

Task 2 prepares a supervised fine-tuning dataset for TacoBot. It demonstrates
the JSONL conversation format used by the later LoRA training task and checks
that each new assistant response is valid JSON with the required fields.

## Execution

The task was run from the repository root with the CPU virtual environment:

```bash
cd /home/irom/fine-tuning-llms
source .venv/bin/activate
python task_2_prepare_data.py
```

The script completed successfully on the laptop without requiring a GPU.

## Data format

Each line in `data/training_data.jsonl` is one JSON object containing a
`messages` array with three roles:

```json
{
  "messages": [
    {"role": "system", "content": "You are TacoBot..."},
    {"role": "user", "content": "What's on the menu?"},
    {"role": "assistant", "content": "{\"response\": \"...\", \"category\": \"menu\"}"}
  ]
}
```

The assistant content is stored as a JSON string inside the outer JSONL
record. This teaches the model to produce structured TacoBot responses.

## Example created

The task creates and validates this conversation:

User message:

```text
Do you have any combo deals?
```

Assistant response:

```json
{
  "response": "Yes! Our Combo #1 includes 2 tacos, chips, and a drink for $8.99!",
  "category": "deals"
}
```

## Validation results

The script confirmed:

- The example contains `system`, `user`, and `assistant` messages.
- The assistant content parses as valid JSON.
- The parsed assistant response contains both `response` and `category`.
- The example was appended to `data/training_data.jsonl`.
- The completion marker was written to `markers/task2_complete.txt`.

The verified run loaded 20 examples and reported 21 examples after the new
record was appended.

## Path fix

The original script referenced the lab-container path
`/root/code/data/training_data.jsonl`. That path caused a permission error on
the laptop. It now resolves the data directory relative to the script, so it
uses this checkout's file:

```text
/home/irom/fine-tuning-llms/data/training_data.jsonl
```

The completion marker also uses a repository-relative path instead of
`/root/markers`.

## Output artifact

```text
data/training_data.jsonl
markers/task2_complete.txt
```

The marker contains:

```text
PREPARE_DATA_COMPLETE
```

## Rerun safety

The task checks whether the combo-deals example already exists before writing.
If it is new, the existing JSONL file is copied to a timestamped path under
`backups/task2/` and the updated file is written atomically. Re-running the
task therefore does not add another copy of the same example.

## What the result shows

Fine-tuning data must be structurally consistent and contain high-quality
target responses. The model learns from the assistant content, so malformed
JSON or inconsistent categories would teach behavior that conflicts with the
TacoBot requirements.
