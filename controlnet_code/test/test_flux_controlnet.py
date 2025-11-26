#!/usr/bin/env python3
"""
Test FLUX.1-Schnell with trained ControlNet, Redux, and LoRA.
Same pipeline as test_flux_redux_lora.py but with ControlNet added.
"""

import argparse
import torch
import cv2
import numpy as np
from PIL import Image
from diffusers import FluxControlNetPipeline, FluxPriorReduxPipeline
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from diffusers.models.controlnets.controlnet_flux import FluxControlNetModel
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
from safetensors.torch import load_file
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict

# Paths
MODEL_DIR = "/home/link/Desktop/Code/fashion gen testing/flux1-schnell"
REDUX_DIR = "/home/link/Desktop/Code/fashion gen testing/flux-redux"
LORA_PATH = "/home/link/Desktop/Code/fashion gen testing/flux_lora_out/lora_flux_rank24_steps1500_step1500.safetensors"
# Single input image (BiRefNet processed) for both Redux and ControlNet
INPUT_IMAGE_PATH = "/home/link/Desktop/Code/fashion gen testing/flux_redux_lora_out/raw_white_bg.png"

# Parameters (matching training)
CAPTION = "professional product photo, garment laid flat on white background, straightened and dewrinkled, studio catalog, centered, evenly lit, sharp edges"
HEIGHT = 512
WIDTH = 512
NUM_INFERENCE_STEPS = 4
GUIDANCE_SCALE = 0.0  # Schnell uses 0
REDUX_SCALE = 0.8
LORA_SCALE = 0.7
LORA_RANK = 24
CONTROLNET_SCALE = 0.2
SEED = 42

device = "cuda"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to ControlNet checkpoint")
    parser.add_argument("--output", type=str, default=None, help="Output path (auto-generated if not specified)")
    args = parser.parse_args()

    CONTROLNET_PATH = args.checkpoint
    if args.output:
        OUTPUT_PATH = args.output
    else:
        import os
        checkpoint_name = os.path.basename(CONTROLNET_PATH)
        OUTPUT_PATH = f"/home/link/Desktop/Code/fashion gen testing/flux_controlnet_out/test_{checkpoint_name}.png"

    # Load input image
    print(f"Loading input image: {INPUT_IMAGE_PATH}")
    input_image = Image.open(INPUT_IMAGE_PATH).convert("RGB")

    # Step 1: Encode text with CLIP + T5 (same as training)
    print("Loading text encoders...")
    tokenizer = CLIPTokenizer.from_pretrained(MODEL_DIR, subfolder="tokenizer")
    tokenizer_2 = T5TokenizerFast.from_pretrained(MODEL_DIR, subfolder="tokenizer_2")
    text_encoder = CLIPTextModel.from_pretrained(MODEL_DIR, subfolder="text_encoder", torch_dtype=torch.bfloat16)
    text_encoder_2 = T5EncoderModel.from_pretrained(MODEL_DIR, subfolder="text_encoder_2", torch_dtype=torch.bfloat16)
    text_encoder.to(device)
    text_encoder_2.to(device)

    print(f"Encoding caption: '{CAPTION}'")
    with torch.no_grad():
        text_inputs = tokenizer([CAPTION], padding="max_length", max_length=77, truncation=True, return_tensors="pt")
        text_inputs_2 = tokenizer_2([CAPTION], padding="max_length", max_length=512, truncation=True, return_tensors="pt")

        prompt_embeds_clip = text_encoder(text_inputs.input_ids.to(device), output_hidden_states=False)
        pooled_prompt_embeds_text = prompt_embeds_clip.pooler_output

        prompt_embeds_t5 = text_encoder_2(text_inputs_2.input_ids.to(device), output_hidden_states=False)[0]

    del text_encoder, text_encoder_2, tokenizer, tokenizer_2
    torch.cuda.empty_cache()

    # Step 2: Encode guidance image with Redux
    print("Loading Redux pipeline...")
    redux_pipe = FluxPriorReduxPipeline.from_pretrained(REDUX_DIR, torch_dtype=torch.bfloat16)
    redux_pipe.to(device)

    print("Encoding guidance image with Redux...")
    with torch.no_grad():
        redux_output = redux_pipe(image=input_image)
        redux_prompt_embeds = redux_output.prompt_embeds
        redux_pooled_embeds = redux_output.pooled_prompt_embeds

    del redux_pipe
    torch.cuda.empty_cache()

    # Step 3: Combine text + image embeddings (with Redux scale)
    print(f"Combining text and image embeddings (Redux scale: {REDUX_SCALE})...")
    redux_prompt_embeds_scaled = redux_prompt_embeds * REDUX_SCALE
    redux_pooled_embeds_scaled = redux_pooled_embeds * REDUX_SCALE

    combined_prompt_embeds = torch.cat([prompt_embeds_t5, redux_prompt_embeds_scaled], dim=1)
    combined_pooled_embeds = pooled_prompt_embeds_text + redux_pooled_embeds_scaled

    print(f"Combined prompt_embeds shape: {combined_prompt_embeds.shape}")
    print(f"Combined pooled_embeds shape: {combined_pooled_embeds.shape}")

    # Step 4: Load transformer with LoRA
    print("Loading transformer...")
    transformer = FluxTransformer2DModel.from_pretrained(
        MODEL_DIR,
        subfolder="transformer",
        torch_dtype=torch.bfloat16
    )

    print(f"Loading LoRA weights (rank={LORA_RANK}, scale={LORA_SCALE})...")
    lora_state_dict = load_file(LORA_PATH)

    scaled_alpha = int(LORA_RANK * LORA_SCALE)
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=scaled_alpha,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    transformer = get_peft_model(transformer, lora_config)
    set_peft_model_state_dict(transformer, lora_state_dict)

    print("Merging LoRA weights...")
    transformer = transformer.merge_and_unload()
    transformer.to(device)

    # Step 5: Load trained ControlNet
    print(f"Loading ControlNet from {CONTROLNET_PATH}...")
    controlnet = FluxControlNetModel.from_pretrained(
        CONTROLNET_PATH,
        torch_dtype=torch.bfloat16
    )

    # Step 6: Load FluxControlNetPipeline
    print("Loading FluxControlNetPipeline...")
    pipe = FluxControlNetPipeline.from_pretrained(
        MODEL_DIR,
        transformer=transformer,
        controlnet=controlnet,
        torch_dtype=torch.bfloat16
    )

    try:
        pipe.enable_sequential_cpu_offload()
    except:
        pipe.to(device)

    # Step 7: Generate with combined text+image conditioning + ControlNet
    print("Generating image...")
    generator = torch.Generator(device=device).manual_seed(SEED)

    result = pipe(
        prompt_embeds=combined_prompt_embeds,
        pooled_prompt_embeds=combined_pooled_embeds,
        control_image=input_image,
        height=HEIGHT,
        width=WIDTH,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        generator=generator,
        controlnet_conditioning_scale=CONTROLNET_SCALE,
    ).images[0]

    result.save(OUTPUT_PATH)
    print(f"Saved output: {OUTPUT_PATH}")
    print("Done!")


if __name__ == "__main__":
    main()
