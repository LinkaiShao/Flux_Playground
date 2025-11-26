#!/usr/bin/env python3
"""
Test FLUX LoRA checkpoints
"""

import os
import torch
from PIL import Image
from diffusers import FluxPipeline
from transformers import AutoModelForImageSegmentation
from torchvision import transforms
from safetensors.torch import load_file, save_file
import tempfile

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
MODEL_DIR = "/home/link/Desktop/Code/fashion gen testing/flux1-schnell"
LORA_DIR = "/home/link/Desktop/Code/fashion gen testing/flux_lora_out"
RAW_IMAGE_PATH = "/home/link/Desktop/Code/fashion gen testing/straighten_image_data/raw/8861882df78e45de96b38e2423fa0fc5.webp"
OUTPUT_DIR = "/home/link/Desktop/Code/fashion gen testing/flux_lora_test"
TEMP_WHITE_BG_PATH = os.path.join(OUTPUT_DIR, "temp_white_bg.png")

# Test multiple checkpoints
LORA_CHECKPOINTS = [
    ("step500", "lora_flux_rank16_steps1500_step500.safetensors"),
    ("step1000", "lora_flux_rank16_steps1500_step1000.safetensors"),
    ("step1500", "lora_flux_rank16_steps1500_step1500.safetensors"),
]

# Generation settings
SEED = 42
NUM_INFERENCE_STEPS = 28
GUIDANCE_SCALE = 3.5
LORA_SCALE = 1.0


def remove_background_birefnet(image_path, output_path):
    """Remove background using BiRefNet and add white background."""
    print("Loading BiRefNet...")
    birefnet = AutoModelForImageSegmentation.from_pretrained(
        "ZhengPeng7/BiRefNet",
        trust_remote_code=True
    )
    birefnet.to(device)

    print(f"Processing image: {image_path}")
    image = Image.open(image_path).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        preds = birefnet(input_tensor)[-1].sigmoid().cpu()

    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(image.size, Image.Resampling.BILINEAR)

    # Create white background
    white_bg = Image.new("RGBA", image.size, (255, 255, 255, 255))
    image_rgba = image.convert("RGBA")
    image_rgba.putalpha(mask)
    white_bg.paste(image_rgba, (0, 0), image_rgba)

    white_bg = white_bg.convert("RGB")
    white_bg.save(output_path)
    print(f"Saved white background image: {output_path}")

    del birefnet
    torch.cuda.empty_cache()

    return white_bg


def load_pipeline_with_lora(model_dir, lora_path):
    """Load Flux pipeline with LoRA."""
    print("\nLoading Flux pipeline...")
    pipe = FluxPipeline.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
    )

    # Memory optimizations
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()

    try:
        pipe.enable_sequential_cpu_offload()
    except:
        pipe.to(device)

    # Load LoRA weights
    print(f"Loading LoRA from: {lora_path}")
    try:
        state_dict = load_file(lora_path)

        # Convert PEFT format to diffusers format
        # Keys are like: transformer.base_model.model.transformer_blocks.X.attn.to_k.lora.down.weight
        # Should be: transformer.transformer_blocks.X.attn.to_k.lora_A.weight
        converted_state_dict = {}
        for key, value in state_dict.items():
            if not key.startswith('transformer.'):
                continue

            # Remove "base_model.model." but keep "transformer." prefix
            new_key = key.replace('transformer.base_model.model.', 'transformer.')

            # Convert lora.down -> lora_A and lora.up -> lora_B
            new_key = new_key.replace('.lora.down.weight', '.lora_A.weight')
            new_key = new_key.replace('.lora.up.weight', '.lora_B.weight')

            converted_state_dict[new_key] = value

        # Save to temporary file and load via diffusers
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_lora_path = os.path.join(tmpdir, "temp_lora.safetensors")
            save_file(converted_state_dict, tmp_lora_path)
            pipe.load_lora_weights(tmpdir, weight_name="temp_lora.safetensors")

        # Fuse LoRA weights
        pipe.fuse_lora(lora_scale=LORA_SCALE)

        print(f"LoRA weights loaded and fused with scale {LORA_SCALE}")
    except Exception as e:
        print(f"Error loading LoRA: {e}")
        import traceback
        traceback.print_exc()
        return None

    return pipe


def generate_straightened_image(pipe, prompt, height=512, width=512, seed=42, num_steps=28, guidance_scale=3.5):
    """Generate straightened garment image."""
    print("\nGenerating straightened image...")
    print(f"  - Steps: {num_steps}")
    print(f"  - Guidance scale: {guidance_scale}")
    print(f"  - LoRA scale: {LORA_SCALE}")
    print(f"  - Seed: {seed}")

    generator = torch.Generator(device=device).manual_seed(seed)

    result = pipe(
        prompt=prompt,
        height=height,
        width=width,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    ).images[0]

    return result


def main():
    print("=" * 60)
    print("Testing FLUX LoRA Checkpoints")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Remove background once (for reference)
    print("\n[Step 1] Removing background...")
    white_bg_image = remove_background_birefnet(RAW_IMAGE_PATH, TEMP_WHITE_BG_PATH)

    # Step 2 & 3: Test each checkpoint
    for checkpoint_name, checkpoint_file in LORA_CHECKPOINTS:
        print(f"\n{'=' * 60}")
        print(f"[Testing {checkpoint_name}]")
        print(f"{'=' * 60}")

        lora_path = os.path.join(LORA_DIR, checkpoint_file)
        output_path = os.path.join(OUTPUT_DIR, f"output_{checkpoint_name}.png")

        # Load pipeline with this checkpoint
        pipe = load_pipeline_with_lora(MODEL_DIR, lora_path)

        if pipe is None:
            print(f"Failed to load {checkpoint_name}. Skipping.")
            continue

        # Generate
        prompt = "professional product photo, garment laid flat on white background, straightened and dewrinkled, studio catalog, centered, evenly lit, sharp edges"

        output_image = generate_straightened_image(
            pipe, prompt,
            height=512, width=512,
            seed=SEED,
            num_steps=NUM_INFERENCE_STEPS,
            guidance_scale=GUIDANCE_SCALE
        )

        # Save
        output_image.save(output_path)
        print(f"\n[Done] Saved: {output_path}")

        # Clean up
        try:
            pipe.unfuse_lora()
        except:
            pass
        del pipe
        torch.cuda.empty_cache()

    print(f"\n{'=' * 60}")
    print("[Done] All checkpoints tested")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
