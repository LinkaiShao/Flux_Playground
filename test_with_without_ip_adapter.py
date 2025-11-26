#!/usr/bin/env python3
"""
Test Flux with and without IP-Adapter image conditioning - side by side comparison
"""

import os
import torch
from PIL import Image
from diffusers import FluxPipeline
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
from safetensors.torch import load_file, save_file
import tempfile

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
MODEL_DIR = "/home/link/Desktop/Code/fashion gen testing/flux1-schnell"
LORA_PATH = "./flux_lora_out/lora_flux_rank16_steps50_step50.safetensors"
GUIDANCE_IMAGE_PATH = "/home/link/Desktop/Code/fashion gen testing/straighten_image_data/on_ground_white_bg/on_ground10.jpg"
OUTPUT_WITH_IP = "./flux_lora_out/test_WITH_ip_adapter_scale10.png"
OUTPUT_WITHOUT_IP = "./flux_lora_out/test_WITHOUT_ip_adapter_scale10.png"

# Settings
IP_SCALE = 10.0  # Try much higher scale
IP_NUM_TOKENS = 4
LORA_SCALE = 0.0  # Test base model only
GUIDANCE_SCALE = 0.0
SEED = 42
NUM_INFERENCE_STEPS = 4
HEIGHT = 512
WIDTH = 512


class ImageProjection(torch.nn.Module):
    """Project CLIP image embeddings to T5 text embedding dimension, expanded to K tokens"""
    def __init__(self, clip_dim: int, text_dim: int, num_tokens: int = 4):
        super().__init__()
        self.num_tokens = num_tokens
        self.proj = torch.nn.Linear(clip_dim, text_dim * num_tokens, bias=False)
        self.text_dim = text_dim

    def forward(self, image_embeds):
        projected = self.proj(image_embeds)
        return projected.view(-1, self.num_tokens, self.text_dim)


def main():
    prompt = "professional product photo, garment laid flat on white background, straightened and dewrinkled, studio catalog, centered, evenly lit, sharp edges"

    print("Loading Flux pipeline...")
    pipe = FluxPipeline.from_pretrained(MODEL_DIR, torch_dtype=torch.bfloat16)
    pipe.enable_attention_slicing()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    try:
        pipe.enable_sequential_cpu_offload()
    except:
        pipe.to(device)

    # Load CLIP image encoder
    print("Loading CLIP image encoder...")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        "openai/clip-vit-large-patch14", torch_dtype=torch.bfloat16
    )
    image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
    image_encoder.requires_grad_(False)
    image_encoder.eval()
    image_encoder.to(device)

    # Load checkpoint
    print(f"Loading checkpoint: {LORA_PATH}")
    state_dict = load_file(LORA_PATH)

    # Separate weights
    lora_state_dict = {}
    image_proj_state_dict = {}

    for key, value in state_dict.items():
        if key.startswith('image_proj.'):
            new_key = key.replace('image_proj.', '')
            image_proj_state_dict[new_key] = value
        elif key.startswith('transformer.'):
            lora_state_dict[key] = value

    print(f"  Image projection keys: {len(image_proj_state_dict)}")

    # Create and load image projection
    clip_projection_dim = image_encoder.config.projection_dim
    t5_hidden_dim = pipe.text_encoder_2.config.d_model

    image_proj = ImageProjection(clip_projection_dim, t5_hidden_dim, num_tokens=IP_NUM_TOKENS)
    image_proj.load_state_dict(image_proj_state_dict)
    image_proj.to(device, dtype=torch.float32)
    image_proj.eval()

    # Encode guidance image
    print(f"Encoding guidance image: {GUIDANCE_IMAGE_PATH}")
    guidance_image = Image.open(GUIDANCE_IMAGE_PATH).convert("RGB")

    with torch.no_grad():
        pixel_values = image_processor(images=guidance_image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device, dtype=torch.bfloat16)
        output = image_encoder(pixel_values)
        image_embeds = output.image_embeds

    generator = torch.Generator(device=device).manual_seed(SEED)

    # Encode text (shared for both)
    with torch.no_grad():
        text_inputs = pipe.tokenizer(
            prompt, padding="max_length", max_length=pipe.tokenizer.model_max_length,
            truncation=True, return_tensors="pt"
        )
        text_input_ids = text_inputs.input_ids.to(device)

        text_inputs_2 = pipe.tokenizer_2(
            prompt, padding="max_length", max_length=pipe.tokenizer_2.model_max_length,
            truncation=True, return_tensors="pt"
        )
        text_input_ids_2 = text_inputs_2.input_ids.to(device)

        prompt_embeds = pipe.text_encoder(text_input_ids)[0].to(torch.bfloat16)
        prompt_embeds_2_base = pipe.text_encoder_2(text_input_ids_2)[0].to(torch.bfloat16)

        pooled_embeds = pipe.text_encoder(text_input_ids, output_hidden_states=True)
        pooled_embeds = pooled_embeds.hidden_states[-2]
        pooled_embeds = pipe.text_encoder.text_model.final_layer_norm(pooled_embeds)
        pooled_embeds = pooled_embeds[:, 0].to(torch.bfloat16)

    # Test 1: WITH IP-Adapter
    print("\n[Test 1] Generating WITH IP-Adapter image conditioning...")
    with torch.no_grad():
        projected_image_embeds = image_proj(image_embeds.to(device, dtype=torch.float32))
        projected_image_embeds = projected_image_embeds * IP_SCALE
        image_tokens = projected_image_embeds.to(torch.bfloat16)
        prompt_embeds_2_with_ip = torch.cat([image_tokens, prompt_embeds_2_base], dim=1)

    generator.manual_seed(SEED)
    result_with_ip = pipe(
        prompt_embeds=prompt_embeds_2_with_ip,
        pooled_prompt_embeds=pooled_embeds,
        height=HEIGHT,
        width=WIDTH,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        generator=generator,
    ).images[0]

    result_with_ip.save(OUTPUT_WITH_IP)
    print(f"Saved WITH IP-Adapter: {OUTPUT_WITH_IP}")

    # Test 2: WITHOUT IP-Adapter
    print("\n[Test 2] Generating WITHOUT IP-Adapter (text only)...")
    generator.manual_seed(SEED)
    result_without_ip = pipe(
        prompt_embeds=prompt_embeds_2_base,  # No image tokens
        pooled_prompt_embeds=pooled_embeds,
        height=HEIGHT,
        width=WIDTH,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        generator=generator,
    ).images[0]

    result_without_ip.save(OUTPUT_WITHOUT_IP)
    print(f"Saved WITHOUT IP-Adapter: {OUTPUT_WITHOUT_IP}")

    print("\n" + "="*60)
    print("Comparison complete!")
    print(f"WITH IP-Adapter:    {OUTPUT_WITH_IP}")
    print(f"WITHOUT IP-Adapter: {OUTPUT_WITHOUT_IP}")
    print("="*60)


if __name__ == "__main__":
    main()
