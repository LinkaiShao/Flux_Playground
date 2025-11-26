#!/usr/bin/env python3
"""
Test Flux IP-Adapter LoRA checkpoints across ranks and guidance scales.
Creates comparison grids for all combinations.
"""

import os
import torch
from PIL import Image, ImageDraw, ImageFont
from diffusers import FluxPipeline
from transformers import AutoModelForImageSegmentation, CLIPVisionModelWithProjection, CLIPImageProcessor
from torchvision import transforms
from safetensors.torch import load_file, save_file
import tempfile
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
MODEL_DIR = "/home/link/Desktop/Code/fashion gen testing/flux1-schnell"
RAW_IMAGE_PATH = "/home/link/Desktop/Code/fashion gen testing/straighten_image_data/raw/8861882df78e45de96b38e2423fa0fc5.webp"
OUTPUT_DIR = "/home/link/Desktop/Code/fashion gen testing/flux_lora_grid_test"
TEMP_WHITE_BG_PATH = os.path.join(OUTPUT_DIR, "temp_white_bg.png")

# Test parameters
RANKS = [32, 24, 16]
CHECKPOINTS = ["step500", "step1000", "step1500"]
GUIDANCE_SCALES = [1.5]  # Clamped to 1.5 as recommended
LORA_SCALES = [0.7, 0.85, 1.0]  # Clamped to 0.7-1.0 range

# IP-Adapter settings
IP_SCALE = 0.6
IP_NUM_TOKENS = 4

# Generation settings
SEED = 42
NUM_INFERENCE_STEPS = 28  # Clamped to 24-30 range (28 is in range)
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


def load_pipeline_with_ip_adapter_lora(model_dir, lora_path, guidance_image_path, lora_scale):
    """Load Flux pipeline with IP-Adapter LoRA."""
    print(f"\nLoading Flux IP-Adapter LoRA: {os.path.basename(lora_path)} (scale={lora_scale})")

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

    # Load CLIP image encoder
    print(f"  Loading CLIP image encoder...")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        "openai/clip-vit-large-patch14",
        torch_dtype=torch.bfloat16
    )
    image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
    image_encoder.requires_grad_(False)
    image_encoder.eval()
    image_encoder.to(device)

    # Load weights
    state_dict = load_file(lora_path)

    # Separate LoRA and image_proj weights
    lora_state_dict = {}
    image_proj_state_dict = {}

    for key, value in state_dict.items():
        if key.startswith('image_proj.'):
            # Remove 'image_proj.' prefix
            new_key = key.replace('image_proj.', '')
            image_proj_state_dict[new_key] = value
        elif key.startswith('transformer.'):
            lora_state_dict[key] = value

    print(f"  LoRA keys: {len(lora_state_dict)}")
    print(f"  Image projection keys: {len(image_proj_state_dict)}")

    # Create image projection layer
    clip_projection_dim = image_encoder.config.projection_dim  # 768
    t5_hidden_dim = pipe.text_encoder_2.config.d_model  # 4096

    image_proj = ImageProjection(clip_projection_dim, t5_hidden_dim, num_tokens=IP_NUM_TOKENS)
    image_proj.load_state_dict(image_proj_state_dict)
    image_proj.to(device, dtype=torch.float32)
    image_proj.eval()

    # Load LoRA to transformer
    if len(lora_state_dict) > 0:
        # Convert PEFT format to diffusers format
        converted_state_dict = {}
        for key, value in lora_state_dict.items():
            # Remove "base_model.model." but keep "transformer." prefix
            new_key = key.replace('transformer.base_model.model.', 'transformer.')
            # Convert lora.down -> lora_A and lora.up -> lora_B
            new_key = new_key.replace('.lora.down.weight', '.lora_A.weight')
            new_key = new_key.replace('.lora.up.weight', '.lora_B.weight')
            converted_state_dict[new_key] = value

        print(f"  Converted LoRA keys: {len(converted_state_dict)}")

        # Save to temporary file and load via diffusers
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_lora_path = os.path.join(tmpdir, "temp_lora.safetensors")
            save_file(converted_state_dict, tmp_lora_path)
            pipe.load_lora_weights(tmpdir, weight_name="temp_lora.safetensors", adapter_name="flux_lora")

        # Set adapter with scale
        pipe.set_adapters(["flux_lora"], adapter_weights=[lora_scale])
        print(f"  LoRA loaded as adapter with scale {lora_scale}")

    # Encode guidance image
    print(f"  Encoding guidance image...")
    guidance_image = Image.open(guidance_image_path).convert("RGB")

    with torch.no_grad():
        pixel_values = image_processor(images=guidance_image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device, dtype=torch.bfloat16)
        output = image_encoder(pixel_values)
        image_embeds = output.image_embeds  # [1, 768]

    return pipe, image_proj, image_encoder, image_embeds


def generate_image_with_ip_adapter(pipe, image_proj, image_embeds, prompt, guidance_scale, seed=42, num_steps=28):
    """Generate image with IP-Adapter conditioning."""
    print(f"  Generating (guidance={guidance_scale})...")

    generator = torch.Generator(device=device).manual_seed(seed)

    # Encode text with both encoders
    with torch.no_grad():
        # CLIP text encoder
        text_inputs = pipe.tokenizer(
            prompt,
            padding="max_length",
            max_length=pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(device)
        prompt_embeds = pipe.text_encoder(text_input_ids)[0].to(torch.bfloat16)

        # T5 text encoder
        text_inputs_2 = pipe.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=pipe.tokenizer_2.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids_2 = text_inputs_2.input_ids.to(device)
        prompt_embeds_2 = pipe.text_encoder_2(text_input_ids_2)[0].to(torch.bfloat16)

        # Pool CLIP embeddings
        pooled_embeds = pipe.text_encoder(text_input_ids, output_hidden_states=True)
        pooled_embeds = pooled_embeds.hidden_states[-2]
        pooled_embeds = pipe.text_encoder.text_model.final_layer_norm(pooled_embeds)
        pooled_embeds = pooled_embeds[:, 0].to(torch.bfloat16)

        # Project image embeddings
        projected_image_embeds = image_proj(image_embeds.to(device, dtype=torch.float32))
        projected_image_embeds = projected_image_embeds * IP_SCALE
        image_tokens = projected_image_embeds.to(torch.bfloat16)

        # Concatenate image tokens with T5 text embeddings
        prompt_embeds_2 = torch.cat([image_tokens, prompt_embeds_2], dim=1)

    # Generate with modified embeddings
    result = pipe(
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_embeds,
        prompt_2_embeds=prompt_embeds_2,
        height=HEIGHT,
        width=WIDTH,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    ).images[0]

    return result


def create_grid(images, labels, rows, cols, output_path, title=""):
    """Create a grid of images with labels."""
    if not images:
        print("No images to create grid")
        return

    # Get dimensions
    img_width, img_height = images[0].size

    # Add space for labels
    label_height = 60
    title_height = 80 if title else 0
    cell_width = img_width
    cell_height = img_height + label_height

    # Create grid
    grid_width = cols * cell_width
    grid_height = title_height + rows * cell_height
    grid = Image.new('RGB', (grid_width, grid_height), color='white')

    draw = ImageDraw.Draw(grid)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 36)
    except:
        font = ImageFont.load_default()
        title_font = font

    # Draw title
    if title:
        bbox = draw.textbbox((0, 0), title, font=title_font)
        text_width = bbox[2] - bbox[0]
        text_x = (grid_width - text_width) // 2
        draw.text((text_x, 20), title, fill='black', font=title_font)

    # Draw images and labels
    for idx, (img, label) in enumerate(zip(images, labels)):
        row = idx // cols
        col = idx % cols

        x = col * cell_width
        y = title_height + row * cell_height + label_height

        grid.paste(img, (x, y))

        # Draw label
        bbox = draw.textbbox((0, 0), label, font=font)
        text_width = bbox[2] - bbox[0]
        text_x = x + (cell_width - text_width) // 2
        text_y = title_height + row * cell_height + 15

        draw.text((text_x, text_y), label, fill='black', font=font)

    grid.save(output_path)
    print(f"Saved grid: {output_path}")


def main():
    print("=" * 60)
    print("Flux LoRA Grid Testing")
    print("=" * 60)
    print(f"Ranks: {RANKS}")
    print(f"Checkpoints: {CHECKPOINTS}")
    print(f"Guidance scales: {GUIDANCE_SCALES}")
    print(f"LoRA scales: {LORA_SCALES}")
    print()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Remove background once
    print("\n[Step 1] Removing background...")
    remove_background_birefnet(RAW_IMAGE_PATH, TEMP_WHITE_BG_PATH)

    prompt = "professional product photo, garment laid flat on white background, straightened and dewrinkled, studio catalog, centered, evenly lit, sharp edges"

    # Test each rank
    for rank in RANKS:
        print(f"\n{'='*60}")
        print(f"Testing Rank {rank}")
        print(f"{'='*60}")

        lora_dir = f"./flux_lora_out_rank{rank}"

        # Test each checkpoint for this rank
        for checkpoint_name in CHECKPOINTS:
            print(f"\n[Checkpoint: {checkpoint_name}]")

            checkpoint_file = f"lora_flux_rank{rank}_steps1500_{checkpoint_name}.safetensors"
            lora_path = os.path.join(lora_dir, checkpoint_file)

            if not os.path.exists(lora_path):
                print(f"  ✗ LoRA file not found: {lora_path}")
                continue

            # Test each LoRA scale
            for lora_scale in LORA_SCALES:
                print(f"\n  [LoRA scale: {lora_scale}]")

                # Load pipeline with IP-Adapter
                pipe, image_proj, image_encoder, image_embeds = load_pipeline_with_ip_adapter_lora(
                    MODEL_DIR, lora_path, TEMP_WHITE_BG_PATH, lora_scale
                )

                # Test different guidance scales
                for guidance_scale in GUIDANCE_SCALES:
                    output_image = generate_image_with_ip_adapter(
                        pipe, image_proj, image_embeds, prompt, guidance_scale,
                        seed=SEED, num_steps=NUM_INFERENCE_STEPS
                    )

                    # Save individual image
                    output_filename = f"rank{rank}_{checkpoint_name}_lora{lora_scale}_guidance{guidance_scale}.png"
                    output_path = os.path.join(OUTPUT_DIR, output_filename)
                    output_image.save(output_path)
                    print(f"    Saved: {output_filename}")

                # Clean up
                if hasattr(pipe, 'unload_lora_weights'):
                    pipe.unload_lora_weights()
                del pipe, image_proj, image_encoder, image_embeds
                torch.cuda.empty_cache()

    # Create mega-grids comparing all combinations at each checkpoint
    print(f"\n{'='*60}")
    print("Creating comparison mega-grids")
    print(f"{'='*60}")

    for checkpoint_name in CHECKPOINTS:
        print(f"\n[Mega-grid for {checkpoint_name}]")

        mega_images = []
        mega_labels = []

        for rank in RANKS:
            for lora_scale in LORA_SCALES:
                for guidance_scale in GUIDANCE_SCALES:
                    img_path = os.path.join(
                        OUTPUT_DIR,
                        f"rank{rank}_{checkpoint_name}_lora{lora_scale}_guidance{guidance_scale}.png"
                    )
                    if os.path.exists(img_path):
                        img = Image.open(img_path)
                        mega_images.append(img)
                        mega_labels.append(f"R{rank} L{lora_scale} G{guidance_scale}")

        if mega_images:
            mega_grid_path = os.path.join(OUTPUT_DIR, f"mega_grid_{checkpoint_name}.png")
            create_grid(
                mega_images, mega_labels,
                rows=len(RANKS) * len(LORA_SCALES), cols=len(GUIDANCE_SCALES),
                output_path=mega_grid_path,
                title=f"All Combinations - {checkpoint_name}"
            )

    print(f"\n{'='*60}")
    print("[Done] Grid testing complete")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Total images: {len(RANKS) * len(CHECKPOINTS) * len(LORA_SCALES) * len(GUIDANCE_SCALES)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
