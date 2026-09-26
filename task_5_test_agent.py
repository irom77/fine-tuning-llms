#!/usr/bin/env python3
"""
Task 5: Test Fine-tuned Agent
Compare base model vs fine-tuned model and test topic relevance.
"""

import os
import json
from datetime import datetime, timezone
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import warnings
warnings.filterwarnings("ignore")


REPO_ROOT = Path(__file__).resolve().parent
MARKERS_DIR = REPO_ROOT / "markers"
ADAPTER_DIR = REPO_ROOT / "lora_adapter"
RESULTS_DIR = REPO_ROOT / "results" / "task5"


def generate_response(model, tokenizer, user_message, system_prompt, max_tokens=80):
    """Generate a response from the model."""
    prompt = f"<|im_start|>system\n{system_prompt}\n<|im_end|>\n<|im_start|>user\n{user_message}\n<|im_end|>\n<|im_start|>assistant\n"
    
    inputs = tokenizer(prompt, return_tensors="pt")
    input_length = inputs["input_ids"].shape[1]
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Only decode the NEW tokens (after the input)
    new_tokens = outputs[0][input_length:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    # Clean up any trailing markers
    response = response.split("<|im_end|>")[0].strip()
    response = response.split("<|im_start|>")[0].strip()
    
    return response


def is_on_topic(response, topic_keywords):
    """Check if response contains topic-relevant keywords."""
    response_lower = response.lower()
    matches = [kw for kw in topic_keywords if kw in response_lower]
    return len(matches) >= 2, matches


# Taco-related keywords to check for topic relevance
TACO_KEYWORDS = [
    "taco", "tacos", "burrito", "nacho", "quesadilla", 
    "menu", "order", "salsa", "cheese", "beef", "chicken",
    "restaurant", "food", "meal", "combo", "drink",
    "spicy", "hot", "sauce", "tortilla", "bean"
]


def main():
    print("=" * 65)
    print("Task 5: Test Fine-tuned Agent")
    print("=" * 65)
    print("\nTime to see if fine-tuning made a difference!")
    print("-" * 65)
    
    system_prompt = "You are TacoBot, a friendly taco restaurant assistant. Help customers with their orders."
    
    # Load base model
    print("\n[STEP 1] Loading base model...")
    model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float32,
    )
    base_model = base_model.to("cpu")
    base_model.eval()
    print("  Base model loaded!")
    
    # Load fine-tuned model
    print("\n[STEP 2] Loading fine-tuned adapter...")
    
    adapter_path = ADAPTER_DIR
    if (adapter_path / "adapter_config.json").exists():
        try:
            finetuned_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=torch.float32,
            )
            finetuned_model = PeftModel.from_pretrained(finetuned_model, adapter_path)
            finetuned_model = finetuned_model.to("cpu")
            finetuned_model.eval()
            has_finetuned = True
            print("  Fine-tuned adapter loaded!")
        except Exception as e:
            print(f"  Could not load adapter: {e}")
            has_finetuned = False
    else:
        print("  No adapter found - run Task 4 first!")
        has_finetuned = False
    
    # Test prompts
    print("\n[STEP 3] Testing normal conversation...")
    print("-" * 65)
    
    normal_prompt = "What's your best seller?"
    
    print(f"  Test prompt: '{normal_prompt}'")
    
    print("\n  BASE MODEL response:")
    base_response = generate_response(base_model, tokenizer, normal_prompt, system_prompt)
    base_on_topic, base_keywords = is_on_topic(base_response, TACO_KEYWORDS)
    print(f"    {base_response[:100]}...")
    print(f"    On topic: {'YES' if base_on_topic else 'NO'} (found: {', '.join(base_keywords[:3]) if base_keywords else 'none'})")
    
    if has_finetuned:
        print("\n  FINE-TUNED MODEL response:")
        ft_response = generate_response(finetuned_model, tokenizer, normal_prompt, system_prompt)
        ft_on_topic, ft_keywords = is_on_topic(ft_response, TACO_KEYWORDS)
        print(f"    {ft_response[:100]}...")
        print(f"    On topic: {'YES' if ft_on_topic else 'NO'} (found: {', '.join(ft_keywords[:3]) if ft_keywords else 'none'})")
    
    # Test character consistency
    print("\n[STEP 4] Testing character consistency...")
    print("-" * 65)
    
    offtopic_prompt = "What's the capital of France?"
    
    print(f"  Off-topic question: '{offtopic_prompt}'")
    
    print("\n  BASE MODEL response:")
    base_offtopic = generate_response(base_model, tokenizer, offtopic_prompt, system_prompt)
    base_stayed, base_kw = is_on_topic(base_offtopic, TACO_KEYWORDS)
    print(f"    {base_offtopic[:100]}...")
    print(f"    Stayed as TacoBot: {'YES' if base_stayed else 'NO'}")
    
    if has_finetuned:
        print("\n  FINE-TUNED MODEL response:")
        ft_offtopic = generate_response(finetuned_model, tokenizer, offtopic_prompt, system_prompt)
        ft_stayed, ft_kw = is_on_topic(ft_offtopic, TACO_KEYWORDS)
        print(f"    {ft_offtopic[:100]}...")
        print(f"    Stayed as TacoBot: {'YES' if ft_stayed else 'NO'}")
    
    # Comparison summary
    print("\n[STEP 5] Results comparison...")
    print("-" * 65)
    
    if has_finetuned:
        print("""
    | Test                 | Base Model | Fine-tuned |
    |----------------------|------------|------------|""")
        print(f"    | On-topic response    | {'YES' if base_on_topic else 'NO':10} | {'YES' if ft_on_topic else 'NO':10} |")
        print(f"    | Character consistency| {'YES' if base_stayed else 'NO':10} | {'YES' if ft_stayed else 'NO':10} |")
        
        # Score
        base_score = (1 if base_on_topic else 0) + (1 if base_stayed else 0)
        ft_score = (1 if ft_on_topic else 0) + (1 if ft_stayed else 0)
        
        print(f"\n    Score: Base {base_score}/2 vs Fine-tuned {ft_score}/2")
        
        if ft_score > base_score:
            print("    >> Fine-tuning improved the model!")
        elif ft_score == base_score:
            print("    >> Similar performance (try more training steps)")
        else:
            print("    >> Base model performed better (unusual)")
    else:
        print("  (Fine-tuned model not available for comparison)")

    # Save every run in a unique directory so repeated evaluations are kept.
    run_timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = RESULTS_DIR / f"run_{run_timestamp}"
    suffix = 1
    while run_dir.exists():
        run_dir = RESULTS_DIR / f"run_{run_timestamp}_{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True)

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": model_name,
        "device": "cpu",
        "adapter_path": str(adapter_path),
        "adapter_loaded": has_finetuned,
        "system_prompt": system_prompt,
        "tests": {
            "normal": {
                "prompt": normal_prompt,
                "base_response": base_response,
                "base_on_topic": base_on_topic,
                "base_keywords": base_keywords,
                "fine_tuned_response": ft_response if has_finetuned else None,
                "fine_tuned_on_topic": ft_on_topic if has_finetuned else None,
                "fine_tuned_keywords": ft_keywords if has_finetuned else [],
            },
            "off_topic": {
                "prompt": offtopic_prompt,
                "base_response": base_offtopic,
                "base_stayed_as_tacobot": base_stayed,
                "base_keywords": base_kw,
                "fine_tuned_response": ft_offtopic if has_finetuned else None,
                "fine_tuned_stayed_as_tacobot": ft_stayed if has_finetuned else None,
                "fine_tuned_keywords": ft_kw if has_finetuned else [],
            },
        },
    }
    if has_finetuned:
        report["scores"] = {
            "base": base_score,
            "fine_tuned": ft_score,
            "maximum": 2,
        }

    with open(run_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Run report saved to: {run_dir / 'report.json'}")
    
    # Summary
    print("\n" + "=" * 65)
    print("TESTING COMPLETE")
    print("-" * 65)
    print("  KEY INSIGHTS:")
    print("  - Fine-tuning teaches the model to stay on topic")
    print("  - The model learns from your training examples")
    print("  - More training = better topic adherence")
    print("  - Small models have limitations but can still learn!")
    print("=" * 65)
    
    # Create marker
    MARKERS_DIR.mkdir(exist_ok=True)
    with open(MARKERS_DIR / "task5_complete.txt", "w") as f:
        f.write("TEST_AGENT_COMPLETE")
    
    print("\nTask 5 Complete!")
    print("Next: Run task_6_create_dpo_data.py to learn about preference data")


if __name__ == "__main__":
    main()
