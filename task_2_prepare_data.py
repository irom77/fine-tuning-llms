#!/usr/bin/env python3
"""
Task 2: Prepare Training Data
Create and validate training data for fine-tuning.
"""

import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data"
MARKERS_DIR = REPO_ROOT / "markers"
BACKUPS_DIR = REPO_ROOT / "backups" / "task2"


def unique_timestamped_path(directory, prefix, suffix):
    """Return a new timestamped path without overwriting an existing backup."""
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    candidate = directory / f"{prefix}_{timestamp}{suffix}"
    counter = 1
    while candidate.exists():
        candidate = directory / f"{prefix}_{timestamp}_{counter}{suffix}"
        counter += 1
    return candidate


def main():
    print("=" * 65)
    print("Task 2: Prepare Training Data")
    print("=" * 65)
    print("\nFine-tuning needs properly formatted training data.")
    print("-" * 65)
    
    # Show the required format
    print("\n[STEP 1] Understanding the training data format...")
    print("-" * 65)
    print("""
    Fine-tuning data uses JSONL format (one JSON per line).
    Each line contains a conversation with messages:
    
    {
      "messages": [
        {"role": "system", "content": "You are TacoBot..."},
        {"role": "user", "content": "What's on the menu?"},
        {"role": "assistant", "content": "{\\"response\\": \\"...\\"...}"}
      ]
    }
    
    Key points:
    - System message defines the agent's behavior
    - User message is the input
    - Assistant message is what we want the model to learn
    """)
    
    # Load existing data
    print("\n[STEP 2] Loading existing training data...")
    
    data_path = DATA_DIR / "training_data.jsonl"
    examples = []
    
    with data_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    
    print(f"  Loaded {len(examples)} training examples")
    
    # Show a sample
    print("\n[STEP 3] Examining a training example...")
    print("-" * 65)
    
    sample = examples[0]
    print("  Sample conversation:")
    for msg in sample["messages"]:
        role = msg["role"].upper()
        content = msg["content"][:60] + "..." if len(msg["content"]) > 60 else msg["content"]
        print(f"    [{role}]: {content}")
    
    print("\n[STEP 4] Creating a NEW training example...")
    print("-" * 65)
    
    new_example = {
        "messages": [
            {
                "role": "system",
                "content": "You are TacoBot, a friendly taco restaurant assistant. Always respond in valid JSON format with 'response' and 'category' keys."
            },
            {
                "role": "user",
                "content": "Do you have any combo deals?"
            },
            {
                "role": "assistant", 
                "content": json.dumps({
                    "response": "Yes! Our Combo #1 includes 2 tacos, chips, and a drink for $8.99!",
                    "category": "deals"
                })
            }
        ]
    }
    
    print("  Your new example:")
    for msg in new_example["messages"]:
        role = msg["role"].upper()
        content = str(msg["content"])[:60]
        print(f"    [{role}]: {content}...")
    
    # Validate the new example
    print("\n[STEP 5] Validating your training example...")
    print("-" * 65)
    
    validation_passed = True
    
    # Check 1: Has all required roles
    roles = [m["role"] for m in new_example["messages"]]
    if "system" in roles and "user" in roles and "assistant" in roles:
        print("  ✓ Has system, user, and assistant messages")
    else:
        print("  ✗ Missing required roles")
        validation_passed = False
    
    # Check 2: Assistant response is valid JSON
    assistant_content = new_example["messages"][2]["content"]
    try:
        parsed = json.loads(assistant_content)
        if "response" in parsed and "category" in parsed:
            print("  ✓ Assistant response is valid JSON with required keys")
        else:
            print("  ✗ Assistant response missing 'response' or 'category' key")
            validation_passed = False
    except json.JSONDecodeError:
        print("  ✗ Assistant response is not valid JSON")
        validation_passed = False
    except TypeError:
        print("  ✗ Assistant response not set (still ___)")
        validation_passed = False
    
    print("\n[STEP 6] Adding your example to training data...")
    print("-" * 65)

    example_already_exists = new_example in examples
    example_added = False

    if validation_passed:
        if example_already_exists:
            new_count = len(examples)
            print("  ~ Example already exists; no duplicate was added")
        else:
            backup_path = unique_timestamped_path(BACKUPS_DIR, "training_data", ".jsonl")
            shutil.copy2(data_path, backup_path)

            updated_examples = examples + [new_example]
            with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", dir=DATA_DIR, delete=False
            ) as temp_file:
                for example in updated_examples:
                    temp_file.write(json.dumps(example) + "\n")
                temp_path = Path(temp_file.name)
            temp_path.replace(data_path)

            new_count = len(updated_examples)
            example_added = True
            print(f"  Backup saved to: {backup_path}")
            print(f"  ✓ Example added! Total examples: {new_count}")
    else:
        new_count = len(examples)
        print("  ✗ Fix validation errors first")
    
    # Summary
    print("\n" + "=" * 65)
    print("TRAINING DATA READY")
    print("-" * 65)
    print(f"  File: {data_path}")
    print(f"  Examples: {new_count}")
    print("  Format: JSONL (one JSON per line)")
    print("\n  KEY INSIGHT:")
    print("  Quality training data is crucial for fine-tuning.")
    print("  The model learns to mimic the assistant responses exactly.")
    print("=" * 65)
    
    # Create marker file
    if validation_passed:
        MARKERS_DIR.mkdir(exist_ok=True)
        with (MARKERS_DIR / "task2_complete.txt").open("w", encoding="utf-8") as f:
            f.write("PREPARE_DATA_COMPLETE")
        
        print("\nTask 2 Complete!")
        print("Next: Run task_3_configure_lora.py to set up LoRA")


if __name__ == "__main__":
    main()
