#!/usr/bin/env python3
"""
Train Flux LoRA with multiple ranks for comparison.
"""

import subprocess
import sys

# Different ranks to test (high to low)
RANKS = [32, 24, 16]

def train_with_rank(rank):
    """Train with specific rank"""
    print("\n" + "="*60)
    print(f"Training Flux LoRA with rank {rank}")
    print("="*60 + "\n")

    output_dir = f"./flux_lora_out_rank{rank}"

    # Run training with command-line args
    cmd = [
        sys.executable, "lora_flux_v1.py",
        "--rank", str(rank),
        "--output_dir", output_dir,
    ]

    try:
        subprocess.run(cmd, check=True)
        print(f"\n✓ Rank {rank} training completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Rank {rank} training failed with error: {e}")
        return False

def main():
    print("="*60)
    print("Flux LoRA Multi-Rank Training")
    print("="*60)
    print(f"Training with ranks: {RANKS}")
    print()

    results = {}

    for rank in RANKS:
        success = train_with_rank(rank)
        results[rank] = "SUCCESS" if success else "FAILED"

    # Summary
    print("\n" + "="*60)
    print("Training Summary")
    print("="*60)
    for rank, status in results.items():
        symbol = "✓" if status == "SUCCESS" else "✗"
        print(f"{symbol} Rank {rank}: {status}")

    print("\nOutput directories:")
    for rank in RANKS:
        print(f"  - ./flux_lora_out_rank{rank}/")

if __name__ == "__main__":
    main()
