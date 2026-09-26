#!/usr/bin/env python3
"""
Task 3: Configure and Apply LoRA
Set up LoRA configuration and apply it to the model.
"""

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
import torch


REPO_ROOT = Path(__file__).resolve().parent
MARKERS_DIR = REPO_ROOT / "markers"
BACKUPS_DIR = REPO_ROOT / "backups" / "task3"


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
    print("Task 3: Configure and Apply LoRA")
    print("=" * 65)
    print("\nLoRA = Low-Rank Adaptation")
    print("Train only 0.3% of parameters instead of 100%!")
    print("-" * 65)
    
    # Load the model
    print("\n[STEP 1] Loading SmolLM2 model...")
    model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float32,  # Use float32 for CPU
    )
    model = model.to("cpu")
    
    # Show original parameters
    original_params = model.num_parameters()
    print(f"  Model loaded: {model_name}")
    print(f"  Total parameters: {original_params:,}")
    
    # Create LoRA config
    print("\n[STEP 2] Creating LoRA configuration...")
    print("-" * 65)
    print("""
    LoRA Key Parameters:
    - r (rank): Size of adapter matrices (higher = more capacity)
    - lora_alpha: Scaling factor (usually 2x rank)
    - target_modules: Which layers to adapt (attention layers)
    """)
    
    lora_rank = 8
    
    lora_alpha = 16
    
    target_modules = ["q_proj", "v_proj"]
    
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    print(f"  LoRA Configuration:")
    print(f"    Rank (r): {lora_config.r}")
    print(f"    Alpha: {lora_config.lora_alpha}")
    print(f"    Target Modules: {lora_config.target_modules}")
    print(f"    Dropout: {lora_config.lora_dropout}")
    
    # Apply LoRA to the model
    print("\n[STEP 3] Applying LoRA to model...")
    print("-" * 65)
    
    model = get_peft_model(model, lora_config)
    
    # Calculate parameter reduction
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    frozen_params = total_params - trainable_params
    
    print(f"  BEFORE LoRA:")
    print(f"    All {original_params:,} params would be trained")
    print(f"\n  AFTER LoRA:")
    print(f"    Frozen (base model): {frozen_params:,}")
    print(f"    Trainable (adapters): {trainable_params:,}")
    print(f"    Percentage trainable: {100 * trainable_params / total_params:.2f}%")
    
    # Show the model structure
    print("\n[STEP 4] Examining LoRA layers...")
    print("-" * 65)
    
    lora_layers = []
    for name, param in model.named_parameters():
        if "lora" in name.lower():
            lora_layers.append((name, param.shape, param.requires_grad))
    
    print(f"  Found {len(lora_layers)} LoRA layers:")
    for name, shape, trainable in lora_layers[:4]:
        short_name = name.split(".")[-2] + "." + name.split(".")[-1]
        print(f"    {short_name}: {list(shape)} (trainable: {trainable})")
    if len(lora_layers) > 4:
        print(f"    ... and {len(lora_layers) - 4} more")
    
    # Memory comparison
    print("\n[STEP 5] Memory savings calculation...")
    print("-" * 65)
    
    full_memory_mb = (original_params * 4 * 3) / (1024 ** 2)  # FP32 + gradients + optimizer
    lora_memory_mb = (trainable_params * 4 * 3) / (1024 ** 2)
    
    print(f"  Full fine-tuning would need: ~{full_memory_mb:.0f} MB")
    print(f"  LoRA fine-tuning needs: ~{lora_memory_mb:.0f} MB")
    print(f"  Memory savings: {100 * (1 - lora_memory_mb/full_memory_mb):.0f}%")
    
    # Save config for next task
    print("\n[STEP 6] Saving configuration for Task 4...")
    
    config_info = {
        "rank": lora_config.r,
        "alpha": lora_config.lora_alpha,
        "target_modules": list(lora_config.target_modules),
        "trainable_params": trainable_params,
        "total_params": total_params,
    }
    
    MARKERS_DIR.mkdir(exist_ok=True)
    config_path = MARKERS_DIR / "lora_config.json"
    if config_path.exists():
        backup_path = unique_timestamped_path(BACKUPS_DIR, "lora_config", ".json")
        shutil.copy2(config_path, backup_path)
        print(f"  Previous configuration backed up to {backup_path}")

    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config_info, f, indent=2)

    print(f"  Configuration saved to {config_path}")
    
    # Summary
    print("\n" + "=" * 65)
    print("LoRA CONFIGURED AND APPLIED")
    print("-" * 65)
    print(f"  Original params: {original_params:,}")
    print(f"  Trainable params: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    print(f"  Memory reduction: ~{100 * (1 - lora_memory_mb/full_memory_mb):.0f}%")
    print("\n  KEY INSIGHT:")
    print("  LoRA freezes the base model and only trains small adapters.")
    print("  This makes fine-tuning possible on consumer hardware!")
    print("=" * 65)
    
    # Create marker file
    with (MARKERS_DIR / "task3_complete.txt").open("w", encoding="utf-8") as f:
        f.write("CONFIGURE_LORA_COMPLETE")
    
    print("\nTask 3 Complete!")
    print("Next: Run task_4_train_lora.py to train the model")


if __name__ == "__main__":
    main()
