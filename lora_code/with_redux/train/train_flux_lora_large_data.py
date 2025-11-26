#!/usr/bin/env python3
"""
Train Flux LoRA on large dataset (grailed + straighten_image_data combined).
Trains with rank 24 only.
Configured for 1000 steps, checkpointing every 250 steps.
Uses lora_flux_redux_v4.py with proper img_ids, sigma-weighted loss, fixed alpha=16.
"""

import subprocess
import sys

# Ranks to train
RANKS = [24]

def train_with_rank(rank):
    """Train with specific rank"""
    print("\n" + "="*60)
    print(f"Training Flux LoRA with rank {rank}")
    print("="*60)
    print("Configuration:")
    print(f"  - LoRA rank: {rank}")
    print("  - Redux scale: 0.8")
    print("  - Total steps: 1000")
    print("  - Checkpoint interval: 250 steps")
    print("  - Dataset: straighten_image_data + grailed_crawl")
    print("  - Memory: Pre-encode all images, store in CPU RAM")
    print("  - Learning rate: 5e-6")
    print("  - LoRA alpha: 16 (fixed)")
    print("  - Version: V4 (proper img_ids, sigma-weighted loss)")
    print("="*60 + "\n")

    output_dir = f"./flux_lora_out/large_data_rank{rank}"

    # Run training with specified parameters (using V4)
    cmd = [
        sys.executable, "lora_flux_redux_v4.py",
        "--rank", str(rank),
        "--redux_scale", "0.8",
        "--lr", "5e-6",
        "--max_train_steps", "1000",
        "--checkpointing_steps", "250",
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
    print("Flux LoRA Multi-Rank Training - Large Dataset (V4)")
    print("="*60)
    print(f"Training with ranks: {RANKS}")
    print("="*60 + "\n")

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
        print(f"  - ./flux_lora_out/large_data_rank{rank}/")

    # Return success if all succeeded
    return all(status == "SUCCESS" for status in results.values())

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
