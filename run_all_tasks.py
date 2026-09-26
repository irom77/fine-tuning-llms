#!/usr/bin/env python3
"""Run the complete fine-tuning lab from a clean generated state."""

import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
VENV_DIR = REPO_ROOT / ".venv"
VENV_PYTHON = VENV_DIR / "bin" / "python"
MARKERS_DIR = REPO_ROOT / "markers"
BACKUPS_DIR = REPO_ROOT / "backups" / "all_tasks"

REQUIRED_PACKAGES = ("torch", "transformers", "peft", "datasets", "accelerate")

TASKS = [
    ("Environment verification", "verify_environment.py", "environment_verified.txt"),
    ("Task 1: Prompt jailbreak", "task_1_prompt_jailbreak.py", "task1_complete.txt"),
    ("Task 2: Prepare training data", "task_2_prepare_data.py", "task2_complete.txt"),
    ("Task 3: Configure LoRA", "task_3_configure_lora.py", "task3_complete.txt"),
    ("Task 4: Train LoRA", "task_4_train_lora.py", "task4_complete.txt"),
    ("Task 5: Test agent", "task_5_test_agent.py", "task5_complete.txt"),
    ("Task 6: Create DPO data", "task_6_create_dpo_data.py", "task6_complete.txt"),
]


def required_packages_ready(python_executable):
    """Return whether all packages required by the lab import successfully."""
    check_script = f"""
import importlib.util
packages = {REQUIRED_PACKAGES!r}
missing = [name for name in packages if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(','.join(missing))
"""
    result = subprocess.run(
        [str(python_executable), "-c", check_script],
        cwd=REPO_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def setup_environment():
    """Create the virtual environment and install packages when necessary."""
    if not VENV_PYTHON.exists():
        print(f"[SETUP] Creating virtual environment: {VENV_DIR}")
        subprocess.run([sys.executable, "-m", "venv", str(VENV_DIR)], check=True)

    if required_packages_ready(VENV_PYTHON):
        print("[SETUP] Virtual environment is ready.")
        return

    pip = VENV_DIR / "bin" / "pip"
    print("[SETUP] Installing required Python packages...")
    subprocess.run([str(pip), "install", "--upgrade", "pip"], check=True)

    if shutil.which("nvidia-smi"):
        torch_install = [str(pip), "install", "torch"]
        print("[SETUP] NVIDIA tooling detected; installing standard PyTorch.")
    else:
        torch_install = [
            str(pip),
            "install",
            "torch",
            "--index-url",
            "https://download.pytorch.org/whl/cpu",
        ]
        print("[SETUP] No NVIDIA tooling detected; installing CPU-only PyTorch.")
    subprocess.run(torch_install, check=True)
    subprocess.run(
        [str(pip), "install", "transformers", "peft", "datasets", "accelerate"],
        check=True,
    )

    if not required_packages_ready(VENV_PYTHON):
        raise RuntimeError("Environment setup finished, but required imports still fail.")
    print("[SETUP] Required packages installed.")


def relaunch_with_virtual_environment():
    """Run this script again with the prepared virtual-environment Python."""
    current_python = Path(sys.executable).resolve()
    if current_python == VENV_PYTHON.resolve():
        return False

    print(f"[SETUP] Relaunching with: {VENV_PYTHON}")
    environment = os.environ.copy()
    environment.setdefault("RUN_ALL_TASKS_STARTED_AT", str(time.time()))
    result = subprocess.run(
        [str(VENV_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]],
        cwd=REPO_ROOT,
        env=environment,
    )
    raise SystemExit(result.returncode)


def unique_backup_directory():
    """Create a timestamped directory without overwriting an older reset."""
    BACKUPS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    candidate = BACKUPS_DIR / f"run_{timestamp}"
    counter = 1
    while candidate.exists():
        candidate = BACKUPS_DIR / f"run_{timestamp}_{counter}"
        counter += 1
    candidate.mkdir()
    return candidate


def reset_generated_state():
    """Preserve and remove generated state before starting the lab."""
    paths = [
        MARKERS_DIR,
        REPO_ROOT / "lora_output",
        REPO_ROOT / "lora_adapter",
        REPO_ROOT / "results" / "task5",
    ]
    existing_paths = [path for path in paths if path.exists()]
    if not existing_paths:
        print("[RESET] No previous generated state found.")
        return None

    backup_dir = unique_backup_directory()
    for path in existing_paths:
        shutil.move(str(path), str(backup_dir / path.name))
    print(f"[RESET] Previous generated state moved to: {backup_dir}")
    return backup_dir


def format_elapsed(seconds):
    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def main():
    started_at = float(os.environ.get("RUN_ALL_TASKS_STARTED_AT", time.time()))
    try:
        setup_environment()
        relaunch_with_virtual_environment()
    except KeyboardInterrupt:
        print("\n✗ Environment setup interrupted by user.")
        print(f"Elapsed time: {format_elapsed(time.time() - started_at)}")
        return 130
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"\n✗ Environment setup failed: {error}")
        print(f"Elapsed time: {format_elapsed(time.time() - started_at)}")
        return 1

    print("=" * 70)
    print("Fine-tuning LLMs Lab - Run All Tasks")
    print("=" * 70)

    try:
        reset_generated_state()

        for index, (label, script_name, marker_name) in enumerate(TASKS, start=1):
            script_path = REPO_ROOT / script_name
            marker_path = MARKERS_DIR / marker_name
            print(f"\n[{index}/{len(TASKS)}] {label}")
            print(f"  Running: {script_name}")

            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=REPO_ROOT,
            )
            if result.returncode != 0:
                print(f"\n✗ {label} failed with exit code {result.returncode}.")
                print(f"Elapsed time: {format_elapsed(time.time() - started_at)}")
                return result.returncode

            if not marker_path.exists():
                print(f"\n✗ {label} finished without creating {marker_path}.")
                print("The pipeline stopped because the task did not complete.")
                print(f"Elapsed time: {format_elapsed(time.time() - started_at)}")
                return 1

            print(f"  ✓ Completion marker found: {marker_path}")

    except KeyboardInterrupt:
        print("\n✗ Pipeline interrupted by user.")
        print(f"Elapsed time: {format_elapsed(time.time() - started_at)}")
        return 130

    elapsed = format_elapsed(time.time() - started_at)
    print("\n" + "=" * 70)
    print("ALL TASKS COMPLETED SUCCESSFULLY")
    print(f"Total elapsed time: {elapsed}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
