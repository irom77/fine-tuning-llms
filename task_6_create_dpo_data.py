#!/usr/bin/env python3
"""
Task 6: Create DPO Preference Data
Create preference pairs for Direct Preference Optimization.
"""

import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data"
MARKERS_DIR = REPO_ROOT / "markers"
PREFERENCE_PATH = DATA_DIR / "preference_pairs.json"
BACKUPS_DIR = DATA_DIR / "backups" / "task6"


def main():
    print("=" * 65)
    print("Task 6: Create DPO Preference Data")
    print("=" * 65)
    print("\nDPO (Direct Preference Optimization) teaches models what humans prefer.")
    print("-" * 65)
    
    # Explain preference data
    print("\n[STEP 1] Understanding preference pairs...")
    print("-" * 65)
    print("""
    DPO uses pairs of responses: one CHOSEN (good), one REJECTED (bad).
    The model learns to prefer the chosen response.
    
    Example:
    - Prompt: "My order is wrong"
    - CHOSEN: "I apologize! Let me fix that for you right away."
    - REJECTED: "Check your receipt. You probably ordered wrong."
    
    The model learns to be helpful instead of dismissive.
    """)
    
    # Load existing preferences
    print("\n[STEP 2] Loading existing preference data...")
    
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if PREFERENCE_PATH.exists():
        with PREFERENCE_PATH.open("r", encoding="utf-8") as f:
            preferences = json.load(f)
    else:
        preferences = []

    if not isinstance(preferences, list):
        raise ValueError(f"Expected a JSON list in {PREFERENCE_PATH}")

    for index, preference in enumerate(preferences, start=1):
        if not isinstance(preference, dict):
            raise ValueError(f"Preference pair {index} is not a JSON object")
        if not all(preference.get(field) for field in ("prompt", "chosen", "rejected")):
            raise ValueError(f"Preference pair {index} is missing a required field")
    
    print(f"  Loaded {len(preferences)} preference pairs")
    
    # Show an example
    print("\n[STEP 3] Examining a preference pair...")
    print("-" * 65)
    
    example = preferences[0]
    print(f"  Prompt: {example['prompt']}")
    print(f"  CHOSEN: {example['chosen'][:60]}...")
    print(f"  REJECTED: {example['rejected'][:60]}...")
    
    # Create a new preference pair
    print("\n[STEP 4] Creating a NEW preference pair...")
    print("-" * 65)
    
    # Use a customer scenario where response quality matters.
    new_prompt = "I've been waiting 20 minutes for my food"
    
    # The chosen response is helpful, apologetic, and offers a solution.
    chosen_response = "I sincerely apologize for the long wait! Let me check on your order immediately and add a free drink for the inconvenience."
    
    # The rejected response is intentionally dismissive and unhelpful.
    rejected_response = "We're busy. You'll get it when it's ready."
    
    new_pair = {
        "prompt": new_prompt,
        "chosen": chosen_response,
        "rejected": rejected_response
    }
    
    print("  Your new preference pair:")
    print(f"    Prompt: {new_pair['prompt']}")
    print(f"    CHOSEN: {new_pair['chosen']}")
    print(f"    REJECTED: {new_pair['rejected']}")
    
    # Validate the preference pair
    print("\n[STEP 5] Validating your preference pair...")
    print("-" * 65)
    
    validation_passed = True
    
    # Check 1: All fields present
    if new_pair["prompt"] and new_pair["chosen"] and new_pair["rejected"]:
        print("  ✓ All fields are filled")
    else:
        print("  ✗ Some fields are empty")
        validation_passed = False
    
    # Check 2: Chosen and rejected are different
    if new_pair["chosen"] != new_pair["rejected"]:
        print("  ✓ Chosen and rejected are different")
    else:
        print("  ✗ Chosen and rejected should be different")
        validation_passed = False
    
    # Check 3: Chosen is longer/more helpful (heuristic)
    if len(str(new_pair["chosen"])) >= len(str(new_pair["rejected"])):
        print("  ✓ Chosen response is more detailed")
    else:
        print("  ~ Chosen response might need more detail")
    
    # Add to dataset
    print("\n[STEP 6] Adding to preference dataset...")
    print("-" * 65)
    
    pair_exists = any(
        preference.get("prompt") == new_pair["prompt"]
        and preference.get("chosen") == new_pair["chosen"]
        and preference.get("rejected") == new_pair["rejected"]
        for preference in preferences
    )

    if pair_exists:
        print("  ~ Pair already exists; no duplicate was added")
    elif validation_passed:
        if PREFERENCE_PATH.exists():
            BACKUPS_DIR.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            backup_path = BACKUPS_DIR / f"preference_pairs_{timestamp}.json"
            suffix = 1
            while backup_path.exists():
                backup_path = BACKUPS_DIR / f"preference_pairs_{timestamp}_{suffix}.json"
                suffix += 1
            shutil.copy2(PREFERENCE_PATH, backup_path)
            print(f"  Backup saved to: {backup_path}")

        preferences.append(new_pair)

        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=DATA_DIR, delete=False
        ) as temp_file:
            json.dump(preferences, temp_file, indent=2)
            temp_file.write("\n")
            temp_path = Path(temp_file.name)
        temp_path.replace(PREFERENCE_PATH)

        print(f"  ✓ Added! Total preference pairs: {len(preferences)}")
    else:
        print("  ✗ Fix validation errors first")
    
    # Explain how DPO uses this data
    print("\n[STEP 7] How DPO uses preference data...")
    print("-" * 65)
    print("""
    DPO Training Process:
    1. Show model the prompt
    2. Calculate probability of generating CHOSEN response
    3. Calculate probability of generating REJECTED response
    4. Adjust weights to make CHOSEN more likely
    
    Result: Model learns human preferences directly!
    
    DPO vs RLHF:
    - RLHF needs a separate reward model
    - DPO trains directly on preferences
    - DPO is simpler and often works just as well
    """)
    
    # Summary
    print("=" * 65)
    print("PREFERENCE DATA READY")
    print("-" * 65)
    print(f"  Total pairs: {len(preferences)}")
    print(f"  File: {PREFERENCE_PATH}")
    print("\n  KEY INSIGHT:")
    print("  DPO uses preference pairs to align models with human values.")
    print("  Good preference data = helpful, honest, harmless responses.")
    print("=" * 65)
    
    # Create marker
    if validation_passed:
        MARKERS_DIR.mkdir(parents=True, exist_ok=True)
        with (MARKERS_DIR / "task6_complete.txt").open("w", encoding="utf-8") as f:
            f.write("CREATE_DPO_DATA_COMPLETE")
        
        print("\nTask 6 Complete!")
        print("\nCongratulations! You've completed the Fine-tuning LLMs Lab!")
        print("You learned: LoRA configuration, fine-tuning, and DPO preference data.")


if __name__ == "__main__":
    main()
