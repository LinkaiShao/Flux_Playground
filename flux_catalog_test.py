#!/usr/bin/env python3
"""
FLUX Catalog Photo Parameter Testing
Tests multiple parameter combinations to find optimal settings for catalog transformation
"""

import os
import json
from pathlib import Path
from PIL import Image
import torch
from diffusers import FluxImg2ImgPipeline

# Paths
INPUT_IMAGE = Path("u2net_output/birefnet_output_2.png").resolve()
PARAMS_FILE = Path("flux_catalog_params.json").resolve()
OUTPUT_DIR = Path("flux_catalog_output").resolve()

# Memory optimization
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

def load_and_prepare_image(image_path):
    """Load image and properly composite RGBA to RGB on white background"""
    img_original = Image.open(image_path)

    # Properly handle transparency
    if img_original.mode in ("RGBA", "LA"):
        white_bg = Image.new("RGB", img_original.size, (255, 255, 255))
        white_bg.paste(img_original, mask=img_original.split()[-1])
        img = white_bg
    else:
        img = img_original.convert("RGB")

    # Resize for VRAM efficiency
    img = img.resize((896, 896), Image.LANCZOS)
    return img

def run_catalog_transform(pipe, img, param_set, param_name):
    """Run FLUX img2img with specific parameter set"""
    print(f"\n{'='*60}")
    print(f"Testing: {param_name}")
    print(f"Description: {param_set['desc']}")
    print(f"Parameters: strength={param_set['strength']}, guidance={param_set['guidance_scale']}, steps={param_set['steps']}")
    print(f"{'='*60}")

    with torch.inference_mode():
        output = pipe(
            prompt=param_set['prompt'],
            negative_prompt=param_set['negative_prompt'],
            image=img,
            strength=param_set['strength'],
            guidance_scale=param_set['guidance_scale'],
            num_inference_steps=param_set['steps'],
        ).images[0]

    # Save output
    output_path = OUTPUT_DIR / f"{param_name}.png"
    output.save(output_path)

    # Save config
    config_path = OUTPUT_DIR / f"{param_name}_config.txt"
    with open(config_path, 'w') as f:
        f.write(f"Parameter Set: {param_name}\n")
        f.write(f"Description: {param_set['desc']}\n\n")
        f.write(json.dumps(param_set, indent=2))

    print(f"✓ Saved: {output_path.name}")
    return output

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load parameter sets
    print(f"Loading parameter sets from: {PARAMS_FILE}")
    with open(PARAMS_FILE, 'r') as f:
        param_sets = json.load(f)
    print(f"Loaded {len(param_sets)} parameter configurations\n")

    # Load and prepare input image
    print(f"Loading input image: {INPUT_IMAGE}")
    img = load_and_prepare_image(INPUT_IMAGE)
    print(f"Input prepared: {img.mode}, {img.size}\n")

    # Save input for reference
    img.save(OUTPUT_DIR / "00_input.png")
    print(f"Saved input reference: 00_input.png\n")

    # Load FLUX pipeline
    print("Loading FLUX Img2Img pipeline...")
    pipe = FluxImg2ImgPipeline.from_pretrained(
        "./flux1-schnell",
        torch_dtype=torch.float16,
        local_files_only=True
    )

    # Memory optimizations
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()
    try:
        pipe.enable_sequential_cpu_offload()
    except Exception:
        pipe.to("cuda")

    print("Pipeline loaded!\n")

    # Run all parameter sets
    results = {}
    for param_name, param_set in param_sets.items():
        try:
            run_catalog_transform(pipe, img, param_set, param_name)
            results[param_name] = "SUCCESS"
        except Exception as e:
            print(f"✗ ERROR: {e}")
            results[param_name] = f"FAILED: {e}"

    # Summary
    print(f"\n{'='*60}")
    print("CATALOG PARAMETER TEST COMPLETE")
    print(f"{'='*60}")
    for name, status in results.items():
        symbol = "✓" if status == "SUCCESS" else "✗"
        print(f"{symbol} {name}: {status}")

    print(f"\nOutputs saved to: {OUTPUT_DIR}")
    print("Compare all images to find best parameters for your use case!")
    print("\nRecommended comparison order:")
    print("1. catalog_2_balanced - Good starting point")
    print("2. catalog_4_texture_priority - If texture is being lost")
    print("3. catalog_3_strong_correction - If wrinkles remain")
    print("4. catalog_5_high_quality - If quality is most important")

if __name__ == "__main__":
    main()
