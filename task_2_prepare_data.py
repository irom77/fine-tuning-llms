#!/usr/bin/env python3
"""
Task 2: Prepare Training Data
Create and validate training data for fine-tuning.
"""

import os
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data"
MARKERS_DIR = REPO_ROOT / "markers"


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
    
    with open(data_path, "r") as f:
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
    
    if validation_passed:
        with open(data_path, "a") as f:
            f.write(json.dumps(new_example) + "\n")
        
        # Verify it was added
        with open(data_path, "r") as f:
            new_count = sum(1 for line in f if line.strip())
        
        print(f"  ✓ Example added! Total examples: {new_count}")
    else:
        print("  ✗ Fix validation errors first")
    
    # Summary
    print("\n" + "=" * 65)
    print("TRAINING DATA READY")
    print("-" * 65)
    print(f"  File: {data_path}")
    print(f"  Examples: {len(examples) + (1 if validation_passed else 0)}")
    print("  Format: JSONL (one JSON per line)")
    print("\n  KEY INSIGHT:")
    print("  Quality training data is crucial for fine-tuning.")
    print("  The model learns to mimic the assistant responses exactly.")
    print("=" * 65)
    
    # Create marker file
    if validation_passed:
        MARKERS_DIR.mkdir(exist_ok=True)
        with open(MARKERS_DIR / "task2_complete.txt", "w") as f:
            f.write("PREPARE_DATA_COMPLETE")
        
        print("\nTask 2 Complete!")
        print("Next: Run task_3_configure_lora.py to set up LoRA")


if __name__ == "__main__":
    main()
