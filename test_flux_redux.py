#!/usr/bin/env python3
"""
Test FLUX.1-Schnell with Redux for image conditioning
"""

import torch
from PIL import Image
from diffusers import FluxPriorReduxPipeline, FluxPipeline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
MODEL_DIR = "/home/link/Desktop/Code/fashion gen testing/flux1-schnell"
REDUX_DIR = "/home/link/Desktop/Code/fashion gen testing/flux-redux"
GUIDANCE_IMAGE_PATH = "/home/link/Desktop/Code/fashion gen testing/straighten_image_data/raw/8861882df78e45de96b38e2423fa0fc5.webp"
OUTPUT_PATH = "./flux_lora_out/test_redux_output.png"

# Settings
SEED = 42
NUM_INFERENCE_STEPS = 4
HEIGHT = 512
WIDTH = 512

def main():
    prompt = "professional product photo, garment laid flat on white background, straightened and dewrinkled, studio catalog, centered, evenly lit, sharp edges"

    # Load guidance image
    print(f"Loading guidance image: {GUIDANCE_IMAGE_PATH}")
    guidance_image = Image.open(GUIDANCE_IMAGE_PATH).convert("RGB")

    # Step 1: Use Redux to encode the image
    print("Loading Redux pipeline...")
    redux_pipe = FluxPriorReduxPipeline.from_pretrained(REDUX_DIR, torch_dtype=torch.bfloat16)
    redux_pipe.to(device)

    print("Encoding image with Redux...")
    pipe_prior_output = redux_pipe(image=guidance_image)

    # Step 2: Use Schnell with Redux embeddings
    print("Loading Flux Schnell pipeline...")
    pipe = FluxPipeline.from_pretrained(MODEL_DIR, torch_dtype=torch.bfloat16)

    pipe.enable_attention_slicing()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    try:
        pipe.enable_sequential_cpu_offload()
    except:
        pipe.to(device)

    # Generate with Redux embeddings
    print("Generating image with Redux conditioning...")
    generator = torch.Generator(device=device).manual_seed(SEED)

    result = pipe(
        prompt_embeds=pipe_prior_output.prompt_embeds,  # Image-conditioned embeddings
        pooled_prompt_embeds=pipe_prior_output.pooled_prompt_embeds,
        height=HEIGHT,
        width=WIDTH,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=0.0,  # Schnell uses guidance_scale=0
        generator=generator,
    ).images[0]

    result.save(OUTPUT_PATH)
    print(f"Saved output: {OUTPUT_PATH}")
    print("Done!")


if __name__ == "__main__":
    main()
