#!/usr/bin/env python3
"""
Task 4: Train with LoRA
Actually fine-tune the model using LoRA.
"""

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
import torch
import warnings
warnings.filterwarnings("ignore")


REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data"
MARKERS_DIR = REPO_ROOT / "markers"
OUTPUT_DIR = REPO_ROOT / "lora_output"
ADAPTER_DIR = REPO_ROOT / "lora_adapter"
BACKUPS_DIR = REPO_ROOT / "backups" / "task4"


def unique_backup_directory():
    """Create a unique directory for preserving a previous training run."""
    BACKUPS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_dir = BACKUPS_DIR / f"run_{timestamp}"
    counter = 1
    while backup_dir.exists():
        backup_dir = BACKUPS_DIR / f"run_{timestamp}_{counter}"
        counter += 1
    backup_dir.mkdir()
    return backup_dir


def backup_existing_outputs():
    """Move prior Trainer outputs aside before creating a new run."""
    existing_paths = [path for path in (OUTPUT_DIR, ADAPTER_DIR) if path.exists()]
    if not existing_paths:
        return None

    backup_dir = unique_backup_directory()
    for path in existing_paths:
        shutil.move(str(path), str(backup_dir / path.name))
    return backup_dir


def main():
    print("=" * 65)
    print("Task 4: Train with LoRA")
    print("=" * 65)
    print("\nTime to actually train the model!")
    print("-" * 65)
    
    # Load model
    print("\n[STEP 1] Loading SmolLM2 model...")
    model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float32,
    )
    model = model.to("cpu")
    
    print(f"  Model: {model_name}")
    print(f"  Parameters: {model.num_parameters():,}")
    
    # Load LoRA config from Task 3 or use defaults
    print("\n[STEP 2] Applying LoRA configuration...")
    
    config_path = MARKERS_DIR / "lora_config.json"
    if config_path.exists():
        with open(config_path, "r") as f:
            saved_config = json.load(f)
        lora_rank = saved_config.get("rank", 8)
        lora_alpha = saved_config.get("alpha", 16)
        print(f"  Using config from Task 3: rank={lora_rank}, alpha={lora_alpha}")
    else:
        lora_rank = 8
        lora_alpha = 16
        print(f"  Using defaults: rank={lora_rank}, alpha={lora_alpha}")
    
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    model = get_peft_model(model, lora_config)
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    
    # Load training data
    print("\n[STEP 3] Loading training data...")
    
    dataset = load_dataset(
        "json",
        data_files=str(DATA_DIR / "training_data.jsonl"),
        split="train"
    )
    
    print(f"  Examples: {len(dataset)}")
    
    # Tokenize
    def tokenize_function(examples):
        texts = []
        for messages in examples["messages"]:
            text = ""
            for msg in messages:
                text += f"<|im_start|>{msg['role']}\n{msg['content']}\n<|im_end|>\n"
            texts.append(text)
        
        return tokenizer(
            texts,
            truncation=True,
            max_length=256,
            padding="max_length",
        )
    
    tokenized = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
    
    max_steps = 50
    
    learning_rate = 2e-4
    
    print("\n[STEP 4] Setting up training...")
    
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        max_steps=max_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        logging_steps=5,
        save_strategy="no",
        report_to="none",
        dataloader_pin_memory=False,
        disable_tqdm=False,
    )
    
    print(f"  Max steps: {training_args.max_steps}")
    print(f"  Batch size: {training_args.per_device_train_batch_size}")
    print(f"  Learning rate: {training_args.learning_rate}")
    
    # Train
    print("\n[STEP 5] Training with LoRA...")
    print("-" * 65)
    print("  Watch the loss decrease as the model learns!")
    print("")

    previous_outputs_backup = backup_existing_outputs()
    if previous_outputs_backup:
        print(f"  Previous outputs backed up to: {previous_outputs_backup}")
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
    )
    
    result = trainer.train()
    
    print("\n  Training complete!")
    print(f"  Final loss: {result.training_loss:.4f}")
    
    # Save the adapter
    print("\n[STEP 6] Saving LoRA adapter...")
    
    adapter_path = ADAPTER_DIR
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    
    # Calculate adapter size
    adapter_size = sum(
        file_path.stat().st_size
        for file_path in adapter_path.iterdir()
        if file_path.is_file()
    ) / (1024 * 1024)
    
    print(f"  Saved to: {adapter_path}")
    print(f"  Adapter size: {adapter_size:.2f} MB")
    print(f"  (Full model would be ~500 MB)")
    
    # Summary
    print("\n" + "=" * 65)
    print("TRAINING COMPLETE!")
    print("-" * 65)
    print(f"  Steps: {training_args.max_steps}")
    print(f"  Final loss: {result.training_loss:.4f}")
    print(f"  Adapter saved: {adapter_path}")
    print(f"  Adapter size: {adapter_size:.2f} MB")
    print("\n  You just fine-tuned a model on consumer hardware!")
    print("=" * 65)
    
    # Create marker
    MARKERS_DIR.mkdir(exist_ok=True)
    with (MARKERS_DIR / "task4_complete.txt").open("w", encoding="utf-8") as f:
        f.write(f"TRAIN_LORA_COMPLETE\nloss={result.training_loss:.4f}")
    
    print("\nTask 4 Complete!")
    print("Next: Run task_5_test_agent.py to test your fine-tuned model")


if __name__ == "__main__":
    main()
