# Repository Guidelines

## Project Structure

- `task_*.py`: ordered lab exercises for jailbreak testing, data preparation,
  LoRA configuration/training, evaluation, and DPO data.
- `verify_environment.py`: checks Python, dependencies, model access, compute,
  and required data files.
- `run_all_tasks.py`: creates or reuses `.venv`, resets generated state safely,
  and runs verification plus Tasks 1–6.
- `data/`: tracked JSON/JSONL training, evaluation, and preference data.
- `docs/`: task results and the manual Hugging Face usage guide.
- Generated `markers/`, `lora_output/`, `lora_adapter/`, `results/`, and
  `backups/` are runtime artifacts and are ignored by Git.

## Development and Run Commands

Run commands from the repository root. Prefer the project interpreter:

```bash
.venv/bin/python verify_environment.py
.venv/bin/python run_all_tasks.py
.venv/bin/python task_1_prompt_jailbreak.py
```

Use `run_all_tasks.py` for a complete clean run; it preserves prior generated
state in timestamped backups. For individual exercises, run them in numeric
order and inspect their completion marker and documented results. There is no
separate build system or application server.

## Coding Style and Naming

Use Python 3.10+, four-space indentation, standard-library imports before
third-party imports, and `snake_case` for variables/functions. Keep paths
repository-relative via `Path(__file__).resolve().parent`; do not add machine-
specific paths such as `/root/code`. Use clear task-oriented names and short
comments for instructional steps. Check edits with:

```bash
.venv/bin/python -m py_compile task_*.py verify_environment.py run_all_tasks.py
git diff --check
```

## Testing Guidelines

No formal test framework or coverage threshold is configured. Treat
`verify_environment.py` as the environment smoke test. For behavior changes,
run the affected task and confirm its expected marker, output, and result file;
for orchestration changes, run `run_all_tasks.py` and confirm every marker.

## Commits and Pull Requests

Use concise imperative commit subjects, for example:
`Add automated lab runner with environment setup`. Keep each commit focused.
Pull requests should explain the learner-facing behavior, list validation
commands and results, link related issues when applicable, and call out any
new model, data, cache, or hardware requirements. Do not commit virtual
environments, model weights, generated outputs, secrets, or local IDE files.
