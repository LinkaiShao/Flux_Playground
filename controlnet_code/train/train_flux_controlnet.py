#!/usr/bin/env python3
"""
Train ControlNet for FLUX with Redux conditioning.

Configuration:
- FLUX transformer: frozen
- Redux encoder: frozen
- LoRA (rank 24, scale 0.7): frozen
- ControlNet: trainable

Input: 4-channel stacked images (RGB + edges)
Output: straightened images
"""
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from diffusers import FluxTransformer2DModel, FluxControlNetModel
from diffusers.models.autoencoders import AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
from safetensors.torch import load_file
from peft import LoraConfig, get_peft_model
from PIL import Image
import numpy as np
from pathlib import Path
from tqdm import tqdm
import os
import json
from dataclasses import dataclass
import argparse

@dataclass
class Args:
    # Model paths
    model_id: str = "/home/link/Desktop/Code/fashion gen testing/flux1-schnell"
    redux_model_id: str = "black-forest-labs/FLUX.1-Redux-dev"
    lora_path: str = "./flux_lora_out/lora_flux_rank24_steps1500_step1500.safetensors"

    # LoRA config
    lora_rank: int = 24
    lora_scale: float = 0.7

    # Redux config
    redux_scale: float = 0.8

    # Training config
    output_dir: str = "./flux_controlnet_out"
    data_dir: str = "./straighten_image_data"
    resolution: int = 512
    train_batch_size: int = 1
    num_train_epochs: int = 100
    learning_rate: float = 1e-5
    gradient_accumulation_steps: int = 4
    checkpointing_steps: int = 500
    seed: int = 42
    mixed_precision: str = "bf16"

    # Text prompt
    prompt: str = "professional product photo, garment laid flat on white background, straightened and dewrinkled, studio catalog, centered, evenly lit, sharp edges"


class ControlNetDataset(Dataset):
    """Dataset for ControlNet training with stacked inputs"""

    def __init__(self, data_dir, resolution=512):
        self.data_dir = Path(data_dir)
        self.resolution = resolution

        # Collect all stacked-straightened pairs
        self.pairs = []

        # Hanged pairs
        hanged_stacked = self.data_dir / "hanged_white_bg_stacked"
        straightened = self.data_dir / "straightened"

        if hanged_stacked.exists():
            for stacked_path in hanged_stacked.glob("*.npy"):
                # Map hanged_X.npy to straightened_X.jpg
                idx = stacked_path.stem.replace("hanged_", "")
                target_path = straightened / f"straightened_{idx}.jpg"
                if not target_path.exists():
                    target_path = straightened / f"straightened_{idx}.png"
                if target_path.exists():
                    self.pairs.append((stacked_path, target_path))

        # On ground pairs
        on_ground_stacked = self.data_dir / "on_ground_white_bg_stacked"

        if on_ground_stacked.exists():
            for stacked_path in on_ground_stacked.glob("*.npy"):
                # Map on_groundX.npy to straightened_X.jpg
                idx = stacked_path.stem.replace("on_ground", "")
                target_path = straightened / f"straightened_{idx}.jpg"
                if not target_path.exists():
                    target_path = straightened / f"straightened_{idx}.png"
                if target_path.exists():
                    self.pairs.append((stacked_path, target_path))

        print(f"Found {len(self.pairs)} training pairs")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        stacked_path, target_path = self.pairs[idx]

        # Load stacked control image (H, W, 4)
        stacked = np.load(str(stacked_path))

        # Load target image
        target = Image.open(target_path).convert("RGB")

        # Resize both to resolution
        # For stacked: resize each channel
        stacked_resized = []
        for c in range(4):
            channel = Image.fromarray(stacked[:, :, c].astype(np.uint8))
            channel = channel.resize((self.resolution, self.resolution), Image.LANCZOS)
            stacked_resized.append(np.array(channel))
        stacked = np.stack(stacked_resized, axis=0)  # (4, H, W)

        # Normalize to [-1, 1]
        stacked = stacked.astype(np.float32) / 127.5 - 1.0

        # Target image
        target = target.resize((self.resolution, self.resolution), Image.LANCZOS)
        target = np.array(target).transpose(2, 0, 1).astype(np.float32) / 127.5 - 1.0

        # Also get the RGB part for Redux conditioning
        rgb_control = stacked[:3]  # First 3 channels are RGB

        return {
            "control": torch.from_numpy(stacked),  # (4, H, W)
            "rgb_control": torch.from_numpy(rgb_control),  # (3, H, W)
            "target": torch.from_numpy(target),  # (3, H, W)
        }


def encode_prompt(prompt, tokenizer, tokenizer_2, text_encoder, text_encoder_2, device):
    """Encode text prompt using CLIP and T5"""
    # CLIP encoding
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids.to(device)

    with torch.no_grad():
        prompt_embeds = text_encoder(text_input_ids, output_hidden_states=False)
        pooled_prompt_embeds = prompt_embeds.pooler_output

    # T5 encoding
    text_inputs_2 = tokenizer_2(
        prompt,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids_2 = text_inputs_2.input_ids.to(device)

    with torch.no_grad():
        prompt_embeds_2 = text_encoder_2(text_input_ids_2)[0]

    return prompt_embeds_2, pooled_prompt_embeds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="./flux_controlnet_out")
    parser.add_argument("--max_train_steps", type=int, default=6000)
    parser.add_argument("--checkpointing_steps", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    cli_args = parser.parse_args()

    args = Args()
    args.output_dir = cli_args.output_dir
    args.checkpointing_steps = cli_args.checkpointing_steps
    args.learning_rate = cli_args.learning_rate
    max_train_steps = cli_args.max_train_steps

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = torch.bfloat16

    print("=" * 60)
    print("FLUX ControlNet Training")
    print("=" * 60)
    print(f"LoRA: rank {args.lora_rank}, scale {args.lora_scale}")
    print(f"Redux scale: {args.redux_scale}")
    print(f"Resolution: {args.resolution}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Output: {args.output_dir}")
    print("=" * 60)

    # Load tokenizers
    print("\nLoading tokenizers...")
    tokenizer = CLIPTokenizer.from_pretrained(args.model_id, subfolder="tokenizer")
    tokenizer_2 = T5TokenizerFast.from_pretrained(args.model_id, subfolder="tokenizer_2")

    # Load text encoders, encode prompt, then free memory
    print("Loading text encoders...")
    text_encoder = CLIPTextModel.from_pretrained(
        args.model_id, subfolder="text_encoder", torch_dtype=weight_dtype
    ).to(device)
    text_encoder_2 = T5EncoderModel.from_pretrained(
        args.model_id, subfolder="text_encoder_2", torch_dtype=weight_dtype
    ).to(device)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)

    # Encode prompt now
    print("Encoding prompt...")
    prompt_embeds, pooled_prompt_embeds = encode_prompt(
        args.prompt, tokenizer, tokenizer_2, text_encoder, text_encoder_2, device
    )

    # Free text encoder memory immediately
    del text_encoder, text_encoder_2
    torch.cuda.empty_cache()
    print(f"GPU memory after freeing text encoders: {torch.cuda.memory_allocated()/1e9:.2f}GB")

    # Load VAE, encode latents, then free memory
    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained(
        args.model_id, subfolder="vae", torch_dtype=weight_dtype
    ).to(device)
    vae.requires_grad_(False)

    # Pre-encode all latents now
    print("Loading dataset and pre-encoding latents...")
    dataset = ControlNetDataset(args.data_dir, args.resolution)

    if len(dataset) == 0:
        print("Error: No training pairs found!")
        return

    encoded_data = []
    for i in range(len(dataset)):
        batch = dataset[i]
        control = batch["control"].unsqueeze(0).to(device, dtype=weight_dtype)
        target = batch["target"].unsqueeze(0).to(device, dtype=weight_dtype)

        with torch.no_grad():
            target_latents = vae.encode(target).latent_dist.sample()
            target_latents = target_latents * vae.config.scaling_factor
            rgb_control = control[:, :3]
            control_latents = vae.encode(rgb_control).latent_dist.sample()
            control_latents = control_latents * vae.config.scaling_factor

        encoded_data.append({
            "target_latents": target_latents.cpu(),
            "control_latents": control_latents.cpu(),
        })
        print(f"  Encoded {i+1}/{len(dataset)}", end="\r")

    print(f"\nEncoded {len(encoded_data)} samples")

    # Free VAE memory
    del vae
    torch.cuda.empty_cache()
    print(f"GPU memory after freeing VAE: {torch.cuda.memory_allocated()/1e9:.2f}GB")

    # Now load FLUX transformer
    print("Loading FLUX transformer...")
    transformer = FluxTransformer2DModel.from_pretrained(
        args.model_id, subfolder="transformer", torch_dtype=weight_dtype
    ).to(device)

    # Apply LoRA with scale
    print(f"Applying LoRA (rank {args.lora_rank}, scale {args.lora_scale})...")
    if os.path.exists(args.lora_path):
        lora_state_dict = load_file(args.lora_path)

        # Configure LoRA with scaled alpha
        scaled_alpha = int(args.lora_rank * args.lora_scale)
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=scaled_alpha,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
        transformer = get_peft_model(transformer, lora_config)

        # Load LoRA weights
        incompatible = transformer.load_state_dict(lora_state_dict, strict=False)
        print(f"LoRA loaded. Missing: {len(incompatible.missing_keys)}, Unexpected: {len(incompatible.unexpected_keys)}")
    else:
        print(f"Warning: LoRA path not found: {args.lora_path}")
        print("Proceeding without LoRA...")

    # Freeze transformer (including LoRA)
    transformer.requires_grad_(False)

    # Load Redux encoder (optional - for future integration)
    # print("Loading Redux encoder...")
    # from transformers import SiglipVisionModel
    # redux_encoder = SiglipVisionModel.from_pretrained(
    #     args.redux_model_id, subfolder="vision_model", torch_dtype=weight_dtype
    # ).to(device)
    # redux_encoder.requires_grad_(False)

    # Enable gradient checkpointing on transformer to save memory
    transformer.enable_gradient_checkpointing()

    # Create ControlNet (medium config)
    print("Creating ControlNet...")
    controlnet = FluxControlNetModel(
        in_channels=64,  # FLUX packed latent channels (16*4 due to patch_size=2)
        num_layers=1,  # Size
        num_single_layers=2,
        attention_head_dim=128,
        num_attention_heads=24,
        joint_attention_dim=4096,
        pooled_projection_dim=768,
        patch_size=2,  # Must match transformer's patch_size
    ).to(device, dtype=weight_dtype)

    # Enable gradient checkpointing on ControlNet
    controlnet.enable_gradient_checkpointing()

    # Only ControlNet is trainable (no projection head needed)
    controlnet.train()

    # Count parameters
    trainable_params = sum(p.numel() for p in controlnet.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in controlnet.parameters())
    print(f"ControlNet parameters: {trainable_params:,} trainable / {total_params:,} total")

    # Optimizer
    optimizer = torch.optim.AdamW(
        controlnet.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.01,
    )

    # Training loop
    print("\nStarting training...")
    global_step = 0
    progress_bar = tqdm(total=max_train_steps, desc="Training")

    import random

    while global_step < max_train_steps:
        # Shuffle data each epoch
        indices = list(range(len(encoded_data)))
        random.shuffle(indices)

        for idx in indices:
            if global_step >= max_train_steps:
                break

            # Get pre-encoded latents
            target_latents = encoded_data[idx]["target_latents"].to(device, dtype=weight_dtype)
            control_latents = encoded_data[idx]["control_latents"].to(device, dtype=weight_dtype)

            # Sample noise and timesteps
            noise = torch.randn_like(target_latents)
            timesteps = torch.randint(
                0, 1000, (target_latents.shape[0],), device=device
            ).long()

            # Add noise to target (flow matching)
            sigmas = (timesteps.float() / 1000).to(weight_dtype)
            sigmas = sigmas.view(-1, 1, 1, 1)
            noisy_latents = (1 - sigmas) * target_latents + sigmas * noise

            # Prepare embeddings
            batch_size = target_latents.shape[0]
            prompt_embeds_batch = prompt_embeds.expand(batch_size, -1, -1)
            pooled_embeds_batch = pooled_prompt_embeds.expand(batch_size, -1)

            # Prepare image IDs for FLUX (packed latent format with patch_size=2)
            latent_h, latent_w = noisy_latents.shape[2], noisy_latents.shape[3]
            num_channels = noisy_latents.shape[1]

            # Create image IDs for packed format (H/2, W/2)
            latent_image_ids = torch.zeros(latent_h // 2, latent_w // 2, 3, device=device, dtype=weight_dtype)
            latent_image_ids[..., 1] = torch.arange(latent_h // 2, device=device, dtype=weight_dtype)[:, None]
            latent_image_ids[..., 2] = torch.arange(latent_w // 2, device=device, dtype=weight_dtype)[None, :]

            # Pack latents using FLUX-style packing (patch_size=2)
            # [B, C, H, W] -> [B, H/2, W/2, C, 2, 2] -> [B, (H/2)*(W/2), C*4]
            packed_noisy_latents = noisy_latents.view(batch_size, num_channels, latent_h // 2, 2, latent_w // 2, 2)
            packed_noisy_latents = packed_noisy_latents.permute(0, 2, 4, 1, 3, 5).contiguous()
            packed_noisy_latents = packed_noisy_latents.view(batch_size, (latent_h // 2) * (latent_w // 2), num_channels * 4)

            packed_control_latents = control_latents.view(batch_size, num_channels, latent_h // 2, 2, latent_w // 2, 2)
            packed_control_latents = packed_control_latents.permute(0, 2, 4, 1, 3, 5).contiguous()
            packed_control_latents = packed_control_latents.view(batch_size, (latent_h // 2) * (latent_w // 2), num_channels * 4)

            packed_image_ids = latent_image_ids.view(-1, 3)  # 2D: [seq_len, 3]

            # Text IDs (2D)
            text_ids = torch.zeros(prompt_embeds_batch.shape[1], 3, device=device, dtype=weight_dtype)

            # Get ControlNet block outputs
            controlnet_block_samples, controlnet_single_block_samples = controlnet(
                hidden_states=packed_noisy_latents,
                controlnet_cond=packed_control_latents,
                timestep=timesteps / 1000,  # Normalize to [0, 1]
                encoder_hidden_states=prompt_embeds_batch,
                pooled_projections=pooled_embeds_batch,
                txt_ids=text_ids,
                img_ids=packed_image_ids,
                return_dict=False,
            )

            # Pass ControlNet outputs through the transformer
            # Gradients flow through transformer (frozen) to train ControlNet
            transformer_output = transformer(
                hidden_states=packed_noisy_latents,
                timestep=timesteps / 1000,
                encoder_hidden_states=prompt_embeds_batch,
                pooled_projections=pooled_embeds_batch,
                txt_ids=text_ids,
                img_ids=packed_image_ids,
                controlnet_block_samples=controlnet_block_samples,
                controlnet_single_block_samples=controlnet_single_block_samples,
                return_dict=False,
            )[0]

            # Compute velocity target (flow matching)
            velocity_target = noise - target_latents
            # Pack velocity target using same FLUX-style packing
            velocity_target_packed = velocity_target.view(batch_size, num_channels, latent_h // 2, 2, latent_w // 2, 2)
            velocity_target_packed = velocity_target_packed.permute(0, 2, 4, 1, 3, 5).contiguous()
            velocity_target_packed = velocity_target_packed.view(batch_size, (latent_h // 2) * (latent_w // 2), num_channels * 4)

            # MSE loss between transformer prediction and target velocity
            loss = F.mse_loss(transformer_output, velocity_target_packed)

            # Backward
            loss.backward()

            if (global_step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            # Logging
            progress_bar.update(1)
            progress_bar.set_postfix(loss=loss.item())

            global_step += 1

            # Checkpointing
            if global_step % args.checkpointing_steps == 0:
                save_path = os.path.join(args.output_dir, f"controlnet_step_{global_step}")
                controlnet.save_pretrained(save_path)
                print(f"\nSaved checkpoint: {save_path}")

    progress_bar.close()
    print(f"\nTraining complete!")

    # Save config
    config = {
        "lora_rank": args.lora_rank,
        "lora_scale": args.lora_scale,
        "redux_scale": args.redux_scale,
        "resolution": args.resolution,
        "prompt": args.prompt,
        "max_train_steps": max_train_steps,
        "learning_rate": args.learning_rate,
    }
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)


if __name__ == "__main__":
    main()
