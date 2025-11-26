#!/usr/bin/env python3
"""
Train multiple LoRA ranks on FLUX with Redux image conditioning.
"""

import os
import subprocess
import sys

RANKS = [64]
MAX_STEPS = 6000
OUTPUT_DIR = "./flux_lora_out"

def train_rank(rank):
    """Train a single rank"""
    print(f"\n{'='*60}")
    print(f"Starting training for rank {rank}")
    print(f"{'='*60}\n")

    cmd = [
        sys.executable,
        "lora_flux_redux.py",
        "--rank", str(rank),
        "--max_train_steps", str(MAX_STEPS),
        "--output_dir", OUTPUT_DIR,
        "--checkpointing_steps", "1000",
    ]

    result = subprocess.run(cmd, cwd=os.getcwd())

    if result.returncode != 0:
        print(f"\n❌ Training failed for rank {rank}")
        return False
    else:
        print(f"\n✓ Training completed for rank {rank}")
        return True

def main():
    print("Training FLUX LoRA with Redux - Multiple Ranks")
    print(f"Ranks: {RANKS}")
    print(f"Steps: {MAX_STEPS}")
    print(f"Output: {OUTPUT_DIR}")

    for rank in RANKS:
        success = train_rank(rank)
        if not success:
            print(f"\nStopping due to error in rank {rank}")
            sys.exit(1)

    print("\n" + "="*60)
    print("All training completed successfully!")
    print("="*60)

if __name__ == "__main__":
    main()
