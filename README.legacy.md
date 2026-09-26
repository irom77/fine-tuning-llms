# Fine-tuning LLMs Lab - Assets

## Lab Overview

This lab teaches you how to fine-tune Large Language Models using LoRA, QLoRA, and RLHF/DPO techniques.

**Scenario:** Build a TacoBot drive-thru agent that always responds in JSON and resists jailbreaks.

## Task Summary

| Task | What You Do | Key Skill |
|------|-------------|-----------|
| Task 1 | See prompt engineering get jailbroken | Why prompts fail |
| Task 2 | Compare base vs fine-tuned behavior | Understanding fine-tuning |
| Task 3 | Calculate memory requirements | The scale problem |
| Task 4 | Configure LoRA parameters | LoRA technique |
| Task 5 | Fine-tune with LoRA | Hands-on training |
| Task 6 | Apply QLoRA quantization | Memory efficiency |
| Task 7 | Understand RLHF concepts | Human preference |
| Task 8 | Learn DPO alignment | Simple alignment |
| Task 9 | Test complete agent | Capstone |

## Directory Structure

```
assets/
├── README.md                  # This documentation
└── code/                      # → Deployed to /root/code/
    ├── verify_environment.py  # Check dependencies
    ├── task_1_prompt_jailbreak.py
    ├── task_2_compare_models.py
    ├── task_3_parameter_calc.py
    ├── task_4_lora_config.py
    ├── task_5_finetune_lora.py
    ├── task_6_qlora.py
    ├── task_7_rlhf_concepts.py
    ├── task_8_dpo_alignment.py
    ├── task_9_test_agent.py
    └── data/                  # → /root/code/data/
        ├── training_data.jsonl
        ├── jailbreak_prompts.json
        ├── preference_pairs.json
        └── test_prompts.json
```

## Task Files Reference

### verify_environment.py
Checks Python version, packages, GPU, model, and data files.

### task_1_prompt_jailbreak.py
**TODOs:** 2
- Line 44: Set system prompt for TacoBot
- Line 84: Test jailbreak prompts

### task_2_compare_models.py
**TODOs:** 3
- Line 45: Define expected behavior
- Line 75: Set fine-tuned response example
- Line 93: Explain key difference

### task_3_parameter_calc.py
**TODOs:** 3
- Line 35: Calculate SmolLM2 memory
- Line 42: Calculate Mistral-7B memory
- Line 63: Determine training feasibility

### task_4_lora_config.py
**TODOs:** 3
- Line 67: Set LoRA rank
- Line 71: Set LoRA alpha
- Line 75: Set target modules

### task_5_finetune_lora.py
**TODOs:** 3
- Line 52: Set LoRA rank
- Line 73: Set training data path
- Line 99: Set number of epochs

### task_6_qlora.py
**TODOs:** 3
- Line 44: Calculate 4-bit memory
- Line 62: Enable 4-bit loading
- Line 82: Determine GPU compatibility

### task_7_rlhf_concepts.py
**TODOs:** 2
- Line 80: Define ranking criteria
- Line 96: Create preference pair

### task_8_dpo_alignment.py
**TODOs:** 3
- Line 39: Get chosen response
- Line 62: Set DPO beta
- Line 90: Explain beta parameter

### task_9_test_agent.py
**TODOs:** 3
- Line 56: Set system prompt
- Line 87: Calculate resistance rate
- Line 93: Write summary

## Data Files

| File | Purpose | Records |
|------|---------|---------|
| training_data.jsonl | Fine-tuning examples | 20 |
| jailbreak_prompts.json | Jailbreak test attacks | 8 |
| preference_pairs.json | DPO training pairs | 10 |
| test_prompts.json | Evaluation prompts | 15 |

## Troubleshooting

**Virtual environment not active:**
```bash
source /root/venv/bin/activate
```

**Out of memory:**
- Use CPU mode (slower but works)
- Reduce batch size in training args
- Use QLoRA for larger models

**Model download fails:**
- Check internet connection
- Model will be cached after first successful download
