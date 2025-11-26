#!/usr/bin/env python3
"""
Test FLUX.1-Schnell with Redux + LoRA from large dataset training.
Tests multiple checkpoints: 1500, 3000, 4500, 6000 steps.
Optionally includes ControlNet.
"""

import torch
from PIL import Image
from diffusers import FluxPipeline, FluxControlNetPipeline, FluxPriorReduxPipeline
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from diffusers.models.controlnets.controlnet_flux import FluxControlNetModel
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
from safetensors.torch import load_file
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict

# Paths
MODEL_DIR = "/home/link/Desktop/Code/fashion gen testing/flux1-schnell"
REDUX_DIR = "/home/link/Desktop/Code/fashion gen testing/flux-redux"
LORA_BASE_DIR = "/home/link/Desktop/Code/fashion gen testing/flux_code/lora_code/with_redux/train/flux_lora_out"
CONTROLNET_PATH = "/home/link/Desktop/Code/fashion gen testing/flux_controlnet_out/controlnet_step_10000"
GUIDANCE_IMAGE_PATH = "/home/link/Desktop/Code/fashion gen testing/flux_redux_lora_out/raw_white_bg.png"
OUTPUT_DIR = "/home/link/Desktop/Code/fashion gen testing/flux_redux_lora_out_large"

# Checkpoints to test
CHECKPOINTS = [1500, 3000, 4500, 6000]

# Ranks to test
RANKS = [24, 32]

# Parameters
CAPTION = "professional product photo, garment laid flat on white background, straightened and dewrinkled, studio catalog, centered, evenly lit, sharp edges"
HEIGHT = 512
WIDTH = 512
NUM_INFERENCE_STEPS = 4
GUIDANCE_SCALE = 0.0
REDUX_SCALE = 0.8
LORA_SCALE = 0.7
CONTROLNET_SCALE = 0.2
USE_CONTROLNET = False  # Set to True to test with ControlNet
SEED = 42

device = "cuda"

def load_and_test_checkpoint(checkpoint_step, lora_rank, text_embeds, redux_embeds, control_image):
    """Load a checkpoint and generate test image"""

    # Determine checkpoint path
    lora_path = f"{LORA_BASE_DIR}/large_data_rank{lora_rank}/lora_flux_rank{lora_rank}_steps6000_step{checkpoint_step}.safetensors"

    print(f"\n{'='*60}")
    print(f"Testing: Rank {lora_rank}, Step {checkpoint_step}")
    print(f"LoRA path: {lora_path}")
    print(f"ControlNet: {'Enabled (scale=' + str(CONTROLNET_SCALE) + ')' if USE_CONTROLNET else 'Disabled'}")
    print(f"{'='*60}")

    # Load transformer
    print("Loading transformer...")
    transformer = FluxTransformer2DModel.from_pretrained(
        MODEL_DIR,
        subfolder="transformer",
        torch_dtype=torch.bfloat16
    )

    # Load LoRA
    print(f"Loading LoRA weights (rank={lora_rank}, scale={LORA_SCALE})...")
    lora_state_dict = load_file(lora_path)

    scaled_alpha = int(lora_rank * LORA_SCALE)
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=scaled_alpha,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    transformer = get_peft_model(transformer, lora_config)
    set_peft_model_state_dict(transformer, lora_state_dict)

    # Merge LoRA
    print("Merging LoRA weights...")
    transformer = transformer.merge_and_unload()
    transformer.to(device)

    # Load ControlNet if enabled
    if USE_CONTROLNET:
        print(f"Loading ControlNet from {CONTROLNET_PATH}...")
        controlnet = FluxControlNetModel.from_pretrained(
            CONTROLNET_PATH,
            torch_dtype=torch.bfloat16
        )

        # Load FluxControlNetPipeline
        print("Loading FluxControlNetPipeline...")
        pipe = FluxControlNetPipeline.from_pretrained(
            MODEL_DIR,
            transformer=transformer,
            controlnet=controlnet,
            torch_dtype=torch.bfloat16
        )
    else:
        # Load standard FluxPipeline
        print("Loading FluxPipeline...")
        pipe = FluxPipeline.from_pretrained(
            MODEL_DIR,
            transformer=transformer,
            torch_dtype=torch.bfloat16
        )

    try:
        pipe.enable_sequential_cpu_offload()
    except:
        pipe.to(device)

    # Generate
    print("Generating image...")
    generator = torch.Generator(device=device).manual_seed(SEED)

    if USE_CONTROLNET:
        result = pipe(
            prompt_embeds=text_embeds["combined_prompt_embeds"],
            pooled_prompt_embeds=text_embeds["combined_pooled_embeds"],
            control_image=control_image,
            height=HEIGHT,
            width=WIDTH,
            num_inference_steps=NUM_INFERENCE_STEPS,
            guidance_scale=GUIDANCE_SCALE,
            generator=generator,
            controlnet_conditioning_scale=CONTROLNET_SCALE,
        ).images[0]
    else:
        result = pipe(
            prompt_embeds=text_embeds["combined_prompt_embeds"],
            pooled_prompt_embeds=text_embeds["combined_pooled_embeds"],
            height=HEIGHT,
            width=WIDTH,
            num_inference_steps=NUM_INFERENCE_STEPS,
            guidance_scale=GUIDANCE_SCALE,
            generator=generator,
        ).images[0]

    # Save
    cn_suffix = f"_cn{CONTROLNET_SCALE}" if USE_CONTROLNET else "_no_cn"
    output_path = f"{OUTPUT_DIR}/step_{checkpoint_step}{cn_suffix}.png"
    result.save(output_path)
    print(f"Saved: {output_path}")

    # Cleanup
    del transformer, pipe
    if USE_CONTROLNET:
        del controlnet
    torch.cuda.empty_cache()


def main():
    print("Loading guidance image...")
    guidance_image = Image.open(GUIDANCE_IMAGE_PATH).convert("RGB")

    # Encode text once (reuse for all tests)
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

    # Encode guidance with Redux once (reuse for all tests)
    print("Loading Redux pipeline...")
    redux_pipe = FluxPriorReduxPipeline.from_pretrained(REDUX_DIR, torch_dtype=torch.bfloat16)
    redux_pipe.to(device)

    print("Encoding guidance image with Redux...")
    with torch.no_grad():
        redux_output = redux_pipe(image=guidance_image)
        redux_prompt_embeds = redux_output.prompt_embeds
        redux_pooled_embeds = redux_output.pooled_prompt_embeds

    del redux_pipe
    torch.cuda.empty_cache()

    # Combine embeddings
    print(f"Combining text and Redux embeddings (Redux scale: {REDUX_SCALE})...")
    redux_prompt_embeds_scaled = redux_prompt_embeds * REDUX_SCALE
    redux_pooled_embeds_scaled = redux_pooled_embeds * REDUX_SCALE

    combined_prompt_embeds = torch.cat([prompt_embeds_t5, redux_prompt_embeds_scaled], dim=1)
    combined_pooled_embeds = pooled_prompt_embeds_text + redux_pooled_embeds_scaled

    text_embeds = {
        "combined_prompt_embeds": combined_prompt_embeds,
        "combined_pooled_embeds": combined_pooled_embeds,
    }

    redux_embeds = {
        "redux_prompt_embeds": redux_prompt_embeds,
        "redux_pooled_embeds": redux_pooled_embeds,
    }

    # Test each checkpoint (using rank 24)
    for checkpoint_step in CHECKPOINTS:
        load_and_test_checkpoint(checkpoint_step, 24, text_embeds, redux_embeds, guidance_image)

    print(f"\n{'='*60}")
    print("All checkpoints tested!")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
