#!/usr/bin/env python3
"""
Test all LoRA ranks with different Redux and LoRA scales.
"""
import torch
from PIL import Image
from diffusers import FluxPipeline, FluxPriorReduxPipeline
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
from safetensors.torch import load_file
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
import os

# Paths
MODEL_DIR = "/home/link/Desktop/Code/fashion gen testing/flux1-schnell"
REDUX_DIR = "/home/link/Desktop/Code/fashion gen testing/flux-redux"
LORA_DIR = "/home/link/Desktop/Code/fashion gen testing/flux_lora_out"
GUIDANCE_IMAGE_PATH = "/home/link/Desktop/Code/fashion gen testing/flux_redux_lora_out/raw_white_bg.png"
OUTPUT_DIR = "/home/link/Desktop/Code/fashion gen testing/flux_redux_lora_out"

# Training caption (same as used during training)
CAPTION = "professional product photo, garment laid flat on white background, straightened and dewrinkled, studio catalog, centered, evenly lit, sharp edges"

# Generation parameters
HEIGHT = 512
WIDTH = 512
NUM_INFERENCE_STEPS = 4
GUIDANCE_SCALE = 0.0  # Schnell uses 0
SEED = 42

device = "cuda"

# Test specific checkpoints for ranks 32 and 64, steps >= 3000
LORA_RANKS = [32, 64]
MIN_STEP = 3000

# LoRA scales to test: 0.5 to 1.3 with 1.0 included
LORA_SCALES = [0.5, 0.7, 0.9, 1.0, 1.1, 1.3]

# Redux scales to test
REDUX_SCALES = [0.3, 0.5, 0.8, 1.0]

def encode_text_and_image(guidance_image_path, caption):
    """Encode text with CLIP+T5 and image with Redux"""
    print("Loading guidance image...")
    guidance_image = Image.open(guidance_image_path).convert("RGB")

    # Step 1: Encode text with CLIP + T5
    print("Loading text encoders...")
    tokenizer = CLIPTokenizer.from_pretrained(MODEL_DIR, subfolder="tokenizer")
    tokenizer_2 = T5TokenizerFast.from_pretrained(MODEL_DIR, subfolder="tokenizer_2")
    text_encoder = CLIPTextModel.from_pretrained(MODEL_DIR, subfolder="text_encoder", torch_dtype=torch.bfloat16).to(device)
    text_encoder_2 = T5EncoderModel.from_pretrained(MODEL_DIR, subfolder="text_encoder_2", torch_dtype=torch.bfloat16).to(device)

    print(f"Encoding caption: '{caption}'")
    text_inputs = tokenizer([caption], padding="max_length", max_length=77, truncation=True, return_tensors="pt")
    text_inputs_2 = tokenizer_2([caption], padding="max_length", max_length=512, truncation=True, return_tensors="pt")

    with torch.no_grad():
        # Encode with CLIP
        prompt_embeds_clip = text_encoder(text_inputs.input_ids.to(device), output_hidden_states=False)
        pooled_prompt_embeds_text = prompt_embeds_clip.pooler_output

        # Encode with T5
        prompt_embeds_t5 = text_encoder_2(text_inputs_2.input_ids.to(device), output_hidden_states=False)[0]

    # Free text encoders
    del text_encoder, text_encoder_2, tokenizer, tokenizer_2
    torch.cuda.empty_cache()

    # Step 2: Encode guidance image with Redux
    print("Loading Redux pipeline...")
    redux_pipe = FluxPriorReduxPipeline.from_pretrained(REDUX_DIR, torch_dtype=torch.bfloat16).to(device)

    print("Encoding guidance image with Redux...")
    with torch.no_grad():
        redux_output = redux_pipe(image=guidance_image)
        redux_prompt_embeds = redux_output.prompt_embeds
        redux_pooled_embeds = redux_output.pooled_prompt_embeds

    # Free Redux
    del redux_pipe
    torch.cuda.empty_cache()

    return {
        "prompt_embeds_t5": prompt_embeds_t5,
        "pooled_prompt_embeds_text": pooled_prompt_embeds_text,
        "redux_prompt_embeds": redux_prompt_embeds,
        "redux_pooled_embeds": redux_pooled_embeds,
    }

def combine_embeddings(embeddings, redux_scale):
    """Combine text and image embeddings with Redux scaling"""
    # Scale Redux embeddings
    redux_prompt_embeds_scaled = embeddings["redux_prompt_embeds"] * redux_scale
    redux_pooled_embeds_scaled = embeddings["redux_pooled_embeds"] * redux_scale

    # Concatenate text and image embeddings
    combined_prompt_embeds = torch.cat([embeddings["prompt_embeds_t5"], redux_prompt_embeds_scaled], dim=1)
    combined_pooled_embeds = embeddings["pooled_prompt_embeds_text"] + redux_pooled_embeds_scaled

    return combined_prompt_embeds, combined_pooled_embeds

def load_lora_transformer(lora_path, rank, lora_scale):
    """Load transformer with LoRA weights"""
    if not os.path.exists(lora_path):
        print(f"Warning: LoRA checkpoint not found: {lora_path}")
        return None

    print(f"Loading transformer from {os.path.basename(lora_path)}...")
    transformer = FluxTransformer2DModel.from_pretrained(
        MODEL_DIR,
        subfolder="transformer",
        torch_dtype=torch.bfloat16
    )

    lora_state_dict = load_file(lora_path)

    # Apply LoRA config with scaled alpha
    scaled_alpha = int(rank * lora_scale)
    print(f"Using LoRA alpha={scaled_alpha} (rank={rank}, effective scale={lora_scale})...")

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=scaled_alpha,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    transformer = get_peft_model(transformer, lora_config)
    set_peft_model_state_dict(transformer, lora_state_dict)

    # Merge LoRA weights
    print("Merging LoRA weights...")
    transformer = transformer.merge_and_unload()
    transformer.to(device)

    return transformer

def generate_image(transformer, combined_prompt_embeds, combined_pooled_embeds, output_path):
    """Generate image with Flux pipeline"""
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

    result.save(output_path)
    print(f"Saved: {output_path}")

    # Free pipeline
    del pipe
    torch.cuda.empty_cache()

def main():
    import glob

    print("=" * 80)
    print("FLUX REDUX LORA CHECKPOINT TESTING (Steps >= 3000)")
    print("=" * 80)

    # Find all checkpoints >= MIN_STEP for specified ranks
    all_checkpoints = []
    for rank in LORA_RANKS:
        pattern = f"{LORA_DIR}/lora_flux_rank{rank}_steps*_step*.safetensors"
        checkpoints = glob.glob(pattern)
        for cp in checkpoints:
            basename = os.path.basename(cp)
            step = int(basename.split("_step")[-1].replace(".safetensors", ""))
            if step >= MIN_STEP:
                all_checkpoints.append((rank, step, cp))

    all_checkpoints.sort(key=lambda x: (x[0], x[1]))

    print(f"\nFound {len(all_checkpoints)} checkpoints:")
    for rank, step, path in all_checkpoints:
        print(f"  - Rank {rank}, Step {step}")

    # Pre-encode text and image (same for all tests)
    print("\nPre-encoding text and image...")
    embeddings = encode_text_and_image(GUIDANCE_IMAGE_PATH, CAPTION)
    print(f"Text embeddings shape: {embeddings['prompt_embeds_t5'].shape}")
    print(f"Redux embeddings shape: {embeddings['redux_prompt_embeds'].shape}")

    # Create subfolders for each rank_step combination
    for rank, step, _ in all_checkpoints:
        checkpoint_dir = os.path.join(OUTPUT_DIR, f"rank{rank}_step{step}")
        os.makedirs(checkpoint_dir, exist_ok=True)

    # Test all combinations
    total_tests = len(all_checkpoints) * len(LORA_SCALES) * len(REDUX_SCALES)
    print(f"\nTesting {total_tests} configurations...")
    print(f"Checkpoints: {len(all_checkpoints)}")
    print(f"LoRA scales: {LORA_SCALES}")
    print(f"Redux scales: {REDUX_SCALES}")

    test_num = 0
    for rank, step, checkpoint_path in all_checkpoints:
        print(f"\n{'=' * 80}")
        print(f"TESTING Rank {rank}, Step {step}")
        print(f"{'=' * 80}")

        for redux_scale in REDUX_SCALES:
            for lora_scale in LORA_SCALES:
                test_num += 1
                print(f"\n--- Test {test_num}/{total_tests}: Rank={rank}, Step={step}, Redux={redux_scale}, LoRA={lora_scale} ---")

                try:
                    # Combine embeddings with current redux scale
                    combined_prompt_embeds, combined_pooled_embeds = combine_embeddings(embeddings, redux_scale)

                    # Load LoRA transformer
                    transformer = load_lora_transformer(checkpoint_path, rank, lora_scale)
                    if transformer is None:
                        print(f"Skipping checkpoint (not found)")
                        break

                    # Generate image
                    output_filename = f"redux{redux_scale}_lora{lora_scale}.png"
                    output_path = os.path.join(OUTPUT_DIR, f"rank{rank}_step{step}", output_filename)
                    generate_image(transformer, combined_prompt_embeds, combined_pooled_embeds, output_path)

                    # Free transformer
                    del transformer
                    torch.cuda.empty_cache()

                except Exception as e:
                    print(f"Error in test {test_num}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

    print("\n" + "=" * 80)
    print("ALL TESTS COMPLETE!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("=" * 80)

if __name__ == "__main__":
    main()
