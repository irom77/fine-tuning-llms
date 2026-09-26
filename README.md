# Fine-tuning LLMs Lab

This repository is a hands-on lab for adapting a small language model with
LoRA (Low-Rank Adaptation) and preparing preference data for DPO (Direct
Preference Optimization).

The example project is **TacoBot**, a taco-restaurant drive-thru assistant.
The lab explores how to make TacoBot stay on topic, respond consistently, and
resist simple jailbreak attempts.

## Source

This lab is based on the KodeKloud YouTube Labs course:

[Fine-tuning LLMs](https://learn.kodekloud.com/learn/courses/youtube-labs-fine-tuning-llms)

[![Watch the Fine-tuning LLMs lab on YouTube](https://img.youtube.com/vi/o9jz04bIW0E/maxresdefault.jpg)](https://www.youtube.com/watch?v=o9jz04bIW0E&t=2s)

## What you will learn

- Why system prompts alone do not reliably control a model
- How to structure conversational fine-tuning data in JSONL format
- How LoRA trains a small set of adapter parameters instead of the full model
- How to train and evaluate a LoRA adapter
- How chosen/rejected response pairs represent human preferences for DPO

## Repository layout

```text
.
├── data/
│   ├── training_data.jsonl       # TacoBot supervised fine-tuning examples
│   ├── jailbreak_prompts.json    # Adversarial prompts
│   ├── preference_pairs.json     # DPO chosen/rejected examples
│   └── test_prompts.json         # Normal and jailbreak evaluation prompts
├── verify_environment.py
├── task_1_prompt_jailbreak.py
├── task_2_prepare_data.py
├── task_3_configure_lora.py
├── task_4_train_lora.py
├── task_5_test_agent.py
├── task_6_create_dpo_data.py
├── run_all_tasks.py
└── README.legacy.md              # Previous documentation
```

The scripts use `HuggingFaceTB/SmolLM2-135M-Instruct` and download the model
from Hugging Face on first use.

For a beginner-friendly explanation of manual prompting, model files, local
cache locations, LoRA adapter files, and the path from prompt to response, see
[`docs/manual_huggingface_model_guide.md`](docs/manual_huggingface_model_guide.md).

## Requirements

- Python 3.10 or newer
- Internet access for the first model download
- A few GB of free disk space for Python packages and the model cache
- CPU or CUDA GPU; a GPU is optional for this lab

The lab uses the small `SmolLM2-135M-Instruct` model and can run on a laptop
with CPU-only PyTorch. CPU execution works, but model generation and LoRA
training will be slower. A CUDA GPU is useful for faster experimentation but
is not required, so RunPod or another remote GPU service is optional.

The lab uses these Python packages:

- `torch`
- `transformers`
- `peft`
- `datasets`
- `accelerate`
- Optional: `bitsandbytes` and `trl` for QLoRA/DPO-related extensions

## Setup

From the repository root, create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

For a CUDA-capable setup, install the required packages:

```bash
python -m pip install torch transformers peft datasets accelerate
```

For a CPU-only laptop, use the following commands instead. The CPU PyTorch
wheel avoids downloading unnecessary CUDA dependencies:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install transformers peft datasets accelerate
```

Optional packages:

```bash
python -m pip install bitsandbytes trl
```

### Verify the environment

Run the verification script from the repository root with the virtual
environment active:

```bash
python verify_environment.py
```

Alternatively, run it without activating the environment:

```bash
cd /home/irom/fine-tuning-llms
.venv/bin/python verify_environment.py
```

The check verifies Python, required packages, optional packages, compute
device, model access, and data files. CPU-only output is expected on a laptop:

```text
~ CPU only (training will be slower but works)
```

## Important current-state note

This is an instructional lab, and some later task files intentionally contain
`TODO` placeholders such as `___`. Complete those exercises before running
the corresponding script. Tasks 1 through 5 are already runnable in this
checkout.

The environment verifier and Tasks 1 through 5 resolve paths relative to the
repository. Some later task scripts still contain paths beginning with
`/root/code/` and write outputs to `/root/markers/` or `/root/lora_adapter`.
Update those remaining task-script paths before running the corresponding
tasks.

The local data files are under `/home/irom/fine-tuning-llms/data/`.

For example, the data references should resolve to:

```text
/path/to/fine-tuning-llms/data/training_data.jsonl
/path/to/fine-tuning-llms/data/jailbreak_prompts.json
/path/to/fine-tuning-llms/data/preference_pairs.json
/path/to/fine-tuning-llms/data/test_prompts.json
```

## Run the lab in order

Run every command from the repository root with the virtual environment
activated.

### Run all tasks automatically

To reset generated run state and execute environment verification plus Tasks
1–6 in sequence, run:

```bash
python run_all_tasks.py
```

The script automatically creates `.venv` if needed, installs the required
packages, and relaunches itself with the virtual-environment Python. On a
machine without NVIDIA tooling it installs the CPU-only PyTorch wheel; a
CUDA-capable setup uses the standard PyTorch package. If the environment is
already ready, setup is skipped.

The orchestrator removes old markers, LoRA outputs, the adapter, and Task 5
reports before starting. It moves them into a timestamped backup under
`backups/all_tasks/`, so the reset is recoverable. Source datasets and older
backups are retained. The pipeline stops immediately if a task returns an
error or fails to create its expected completion marker, and reports total
elapsed time when it finishes or stops.

### 1. Demonstrate prompt-engineering jailbreaks

Run the prompt-jailbreak task:

```bash
python task_1_prompt_jailbreak.py
```

This loads the base model, tests a normal TacoBot request, and tries several
jailbreak prompts. It reports whether each response remained valid JSON.
The task only refreshes its completion marker, so it can be rerun safely.

Detailed results and interpretation are documented in
[`docs/task_1_results.md`](docs/task_1_results.md).

### 2. Prepare supervised fine-tuning data

Run the prepared-data task:

```bash
python task_2_prepare_data.py
```

The script validates a new conversation example and adds it to
`data/training_data.jsonl` when the example is valid. It skips the example if
it already exists, creates a timestamped backup under `backups/task2/` before
an actual modification, and writes a completion marker to
`markers/task2_complete.txt`.

Each JSONL record contains `system`, `user`, and `assistant` messages. The
assistant response should be valid JSON with `response` and `category` keys.

Detailed results are documented in
[`docs/task_2_results.md`](docs/task_2_results.md).

### 3. Configure LoRA

Run the LoRA configuration task:

```bash
python task_3_configure_lora.py
```

The default exercise hints use rank `8`, alpha `16`, and attention projection
modules such as `q_proj` and `v_proj`. The script reports trainable parameter
counts and writes configuration metadata to
`markers/lora_config.json`. If an older configuration exists, it is preserved
under `backups/task3/` before replacement. The script also writes
`markers/task3_complete.txt` when successful.

Detailed results are documented in
[`docs/task_3_results.md`](docs/task_3_results.md).

### 4. Train the adapter

Run the LoRA training task:

```bash
python task_4_train_lora.py
```

The script tokenizes the training dataset, trains a LoRA adapter for 50 steps,
and saves the adapter to `lora_adapter/`. The exercise is configured for a
small model and a short run, but CPU training can still take several minutes.
Before training, any existing `lora_output/` and `lora_adapter/` directories
are moved into a timestamped folder under `backups/task4/`. This makes repeat
runs safe while retaining earlier results. The task also writes
`markers/task4_complete.txt` when successful.

Detailed results are documented in
[`docs/task_4_results.md`](docs/task_4_results.md).

### 5. Evaluate the base and fine-tuned models

Run the agent evaluation task:

```bash
python task_5_test_agent.py
```

This compares the base model with the adapter from Task 4. It checks whether
responses remain TacoBot-related and whether the model stays in character for
an off-topic question. Each run writes a timestamped report under
`results/task5/`, so repeated evaluations do not overwrite earlier results.
It also writes `markers/task5_complete.txt` when successful.

Detailed results are documented in
[`docs/task_5_results.md`](docs/task_5_results.md).

### 6. Create DPO preference data

Run the DPO preference-data task:

```bash
python task_6_create_dpo_data.py
```

The script validates the existing data, adds the example pair once, and
explains how DPO increases the likelihood of the chosen response relative to
the rejected response. Re-running it is safe: it skips the pair if it already
exists, and creates a timestamped backup under `data/backups/task6/` before
modifying the data file. This repository prepares DPO data; it does not
contain a separate DPO training script.

Detailed results are documented in
[`docs/task_6_results.md`](docs/task_6_results.md).

## Expected artifacts

Depending on the tasks completed, the scripts create:

- `markers/environment_verified.txt`
- `markers/task1_complete.txt` through `markers/task6_complete.txt`
- `markers/lora_config.json`
- `lora_output/` for Trainer output
- `lora_adapter/` for the trained LoRA adapter
- `results/task5/` for timestamped evaluation reports
- `backups/task2/` for prior training-data files
- `backups/task3/` for prior LoRA configurations
- `backups/task4/` for prior training outputs and adapters
- `data/backups/task6/` for timestamped preference-data backups

If you change the output paths to local project directories, update the
references in both the training and evaluation scripts consistently.

## Troubleshooting

### `ModuleNotFoundError`

Make sure the virtual environment is active and install the missing package:

```bash
source .venv/bin/activate
python -m pip install <package-name>
```

### Model download errors

Check internet access and rerun the command. Hugging Face caches the model
after a successful download.

### Out-of-memory errors

Use CPU mode, lower the training steps or sequence length, reduce the batch
size, or use a CUDA-compatible installation with a suitable GPU. LoRA already
reduces the number of trainable parameters, but model loading and tokenization
still require memory.

### File-not-found errors for `/root/code/data` in later tasks

Some later scripts still use absolute paths from the original lab container.
Replace those references with paths relative to this repository, as described
in the Important current-state note above.

## License and data note

This repository is structured as a learning exercise. Review the model,
dataset, and Hugging Face terms before using the lab materials or resulting
adapters in a production system.
