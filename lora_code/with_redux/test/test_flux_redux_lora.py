#!/usr/bin/env python3
"""
Test FLUX.1-Schnell with Redux image conditioning and trained LoRA.
IMPORTANT: Must match training pipeline - text encoders + Redux image encoding
"""

import torch
from PIL import Image
from diffusers import FluxPipeline, FluxPriorReduxPipeline
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

# Paths
MODEL_DIR = "/home/link/Desktop/Code/fashion gen testing/flux1-schnell"
REDUX_DIR = "/home/link/Desktop/Code/fashion gen testing/flux-redux"
LORA_PATH = "/home/link/Desktop/Code/fashion gen testing/flux_lora_out/lora_flux_rank16_steps1500_step1500.safetensors"
GUIDANCE_IMAGE_PATH = "/home/link/Desktop/Code/fashion gen testing/straighten_image_data/raw/8861882df78e45de96b38e2423fa0fc5.webp"

# Parameters (matching training)
CAPTION = "professional product photo, garment laid flat on white background, straightened and dewrinkled, studio catalog, centered, evenly lit, sharp edges"
HEIGHT = 512
WIDTH = 512
NUM_INFERENCE_STEPS = 4
GUIDANCE_SCALE = 0.0  # Schnell uses 0
REDUX_SCALE = 0.8  # Scale down Redux image conditioning (0.0-1.0+)
LORA_SCALE = 1.0  # LoRA strength (0.0=off, 1.0=full, >1.0=amplified)
SEED = 42

OUTPUT_PATH = f"/home/link/Desktop/Code/fashion gen testing/flux_lora_out/test_redux_lora_output_scale{LORA_SCALE}_v2.png"

device = "cuda"

def main():
    print(f"Loading guidance image: {GUIDANCE_IMAGE_PATH}")
    guidance_image = Image.open(GUIDANCE_IMAGE_PATH).convert("RGB")

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
        # Tokenize
        text_inputs = tokenizer([CAPTION], padding="max_length", max_length=77, truncation=True, return_tensors="pt")
        text_inputs_2 = tokenizer_2([CAPTION], padding="max_length", max_length=512, truncation=True, return_tensors="pt")

        # Encode with CLIP
        prompt_embeds_clip = text_encoder(text_inputs.input_ids.to(device), output_hidden_states=False)
        pooled_prompt_embeds_text = prompt_embeds_clip.pooler_output  # [1, 768]

        # Encode with T5
        prompt_embeds_t5 = text_encoder_2(text_inputs_2.input_ids.to(device), output_hidden_states=False)[0]  # [1, seq_len, 4096]

    # Free text encoders
    del text_encoder, text_encoder_2, tokenizer, tokenizer_2
    torch.cuda.empty_cache()

    # Step 2: Encode guidance image with Redux
    print("Loading Redux pipeline...")
    redux_pipe = FluxPriorReduxPipeline.from_pretrained(REDUX_DIR, torch_dtype=torch.bfloat16)
    redux_pipe.to(device)

    print("Encoding guidance image with Redux...")
    with torch.no_grad():
        redux_output = redux_pipe(image=guidance_image)
        redux_prompt_embeds = redux_output.prompt_embeds  # [1, N, 4096]
        redux_pooled_embeds = redux_output.pooled_prompt_embeds  # [1, 768]

    # Free Redux
    del redux_pipe
    torch.cuda.empty_cache()

    # Step 3: Combine text + image embeddings (with Redux scale)
    print(f"Combining text and image embeddings (Redux scale: {REDUX_SCALE})...")
    # Scale Redux embeddings
    redux_prompt_embeds_scaled = redux_prompt_embeds * REDUX_SCALE
    redux_pooled_embeds_scaled = redux_pooled_embeds * REDUX_SCALE

    # Concatenate text and image embeddings along sequence dimension
    combined_prompt_embeds = torch.cat([prompt_embeds_t5, redux_prompt_embeds_scaled], dim=1)  # [1, text_seq+img_seq, 4096]
    # Add pooled embeddings
    combined_pooled_embeds = pooled_prompt_embeds_text + redux_pooled_embeds_scaled  # [1, 768]

    print(f"Combined prompt_embeds shape: {combined_prompt_embeds.shape}")
    print(f"Combined pooled_embeds shape: {combined_pooled_embeds.shape}")

    # Step 4: Load transformer with LoRA
    print("Loading transformer...")
    transformer = FluxTransformer2DModel.from_pretrained(
        MODEL_DIR,
        subfolder="transformer",
        torch_dtype=torch.bfloat16
    )

    print(f"Loading LoRA weights from {LORA_PATH}...")
    from safetensors.torch import load_file
    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict

    lora_state_dict = load_file(LORA_PATH)

    # Apply LoRA config with scaled alpha for adjusting LoRA strength
    # LoRA scaling is controlled by: scaling = lora_alpha / r
    # To scale by LORA_SCALE, we adjust lora_alpha
    scaled_alpha = int(16 * LORA_SCALE)
    print(f"Using LoRA alpha={scaled_alpha} (rank=16, effective scale={LORA_SCALE})...")

    lora_config = LoraConfig(
        r=16,
        lora_alpha=scaled_alpha,  # Scale alpha instead of weights
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    transformer = get_peft_model(transformer, lora_config)
    set_peft_model_state_dict(transformer, lora_state_dict)

    # Merge LoRA weights
    print("Merging LoRA weights...")
    transformer = transformer.merge_and_unload()
    transformer.to(device)

    # Step 5: Load Flux pipeline and generate
    print("Loading Flux Schnell pipeline...")
    pipe = FluxPipeline.from_pretrained(
        MODEL_DIR,
        transformer=transformer,
        torch_dtype=torch.bfloat16
    )

    try:
        pipe.enable_sequential_cpu_offload()
    except:
        pipe.to(device)

    # Generate with combined text+image conditioning
    print("Generating image...")
    generator = torch.Generator(device=device).manual_seed(SEED)

    result = pipe(
        prompt_embeds=combined_prompt_embeds,
        pooled_prompt_embeds=combined_pooled_embeds,
        height=HEIGHT,
        width=WIDTH,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        generator=generator,
    ).images[0]

    result.save(OUTPUT_PATH)
    print(f"Saved output: {OUTPUT_PATH}")
    print("Done!")


if __name__ == "__main__":
    main()
