#!/usr/bin/env python3
"""
Verify Environment
Check that all dependencies are installed and model is ready.
"""

import os
import sys


REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(REPO_ROOT, "data")
MARKERS_DIR = os.path.join(REPO_ROOT, "markers")


def main():
    print("=" * 65)
    print("Fine-tuning LLMs Lab - Environment Verification")
    print("=" * 65)
    
    all_passed = True
    
    # Check 1: Python version
    print("\n[CHECK 1] Python Version...")
    py_version = sys.version_info
    if py_version.major >= 3 and py_version.minor >= 10:
        print(f"  ✓ Python {py_version.major}.{py_version.minor} (3.10+ required)")
    else:
        print(f"  ✗ Python {py_version.major}.{py_version.minor} - Need 3.10+")
        all_passed = False
    
    # Check 2: Required packages
    print("\n[CHECK 2] Required Packages...")
    packages = {
        "torch": "PyTorch",
        "transformers": "HuggingFace Transformers",
        "peft": "PEFT (LoRA)",
        "datasets": "Datasets",
        "accelerate": "Accelerate",
    }
    
    for package, name in packages.items():
        try:
            __import__(package)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} - Not installed")
            all_passed = False
    
    # Check 3: Optional packages (for QLoRA and DPO)
    print("\n[CHECK 3] Optional Packages...")
    optional = {
        "bitsandbytes": "BitsAndBytes (QLoRA)",
        "trl": "TRL (DPO/RLHF)",
    }
    
    # Suppress warnings from optional packages
    import warnings
    import contextlib
    import io
    
    for package, name in optional.items():
        try:
            # Suppress stderr output during import
            with contextlib.redirect_stderr(io.StringIO()):
                warnings.filterwarnings("ignore")
                __import__(package)
                warnings.resetwarnings()
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ~ {name} - Not installed (optional)")
    
    # Check 4: GPU/CPU
    print("\n[CHECK 4] Compute Device...")
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  ✓ GPU: {gpu_name} ({gpu_mem:.1f}GB)")
        else:
            print("  ~ CPU only (training will be slower but works)")
    except Exception as e:
        print(f"  ~ Could not detect device: {e}")
    
    # Check 5: Model availability
    print("\n[CHECK 5] Model Files...")
    model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"  ✓ SmolLM2-135M accessible")
    except Exception as e:
        print(f"  ~ Model will be downloaded on first use")
    
    # Check 6: Data files
    print("\n[CHECK 6] Data Files...")
    data_files = [
        os.path.join(DATA_DIR, "training_data.jsonl"),
        os.path.join(DATA_DIR, "jailbreak_prompts.json"),
        os.path.join(DATA_DIR, "preference_pairs.json"),
        os.path.join(DATA_DIR, "test_prompts.json"),
    ]
    
    for filepath in data_files:
        if os.path.exists(filepath):
            print(f"  ✓ {os.path.basename(filepath)}")
        else:
            print(f"  ✗ {os.path.basename(filepath)} - Missing")
            all_passed = False
    
    # Summary
    print("\n" + "=" * 65)
    if all_passed:
        print("Environment Verification: PASSED")
        print("You are ready to start the lab!")
        
        # Create marker file
        os.makedirs(MARKERS_DIR, exist_ok=True)
        with open(os.path.join(MARKERS_DIR, "environment_verified.txt"), "w") as f:
            f.write("ENVIRONMENT_VERIFIED")
    else:
        print("Environment Verification: ISSUES FOUND")
        print("Please resolve the issues above before continuing.")
    print("=" * 65)


if __name__ == "__main__":
    main()
