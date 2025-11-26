#!/usr/bin/env python3
"""
Train LoRA on FLUX with image conditioning from flux-redux.
Similar approach to lora_sdxl_v2 but adapted for FLUX architecture.
"""

import os
import warnings
from dataclasses import dataclass, field
from typing import List, Dict, Any
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
from datasets import Dataset
from torchvision import transforms
from tqdm.auto import tqdm

from diffusers import (
    FluxPipeline,
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
)
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from transformers import (
    CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast,
    CLIPVisionModelWithProjection, CLIPImageProcessor
)

from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration

# ----------------------------
# Args
# ----------------------------
@dataclass
class Args:
    # Model paths
    pretrained_model: str = "/home/link/Desktop/Code/fashion gen testing/flux1-schnell"
    use_redux: bool = False  # Redux image conditioning not yet implemented for Flux LoRA

    # Data
    data_root: str = "/home/link/Desktop/Code/fashion gen testing/straighten_image_data"
    use_ground: bool = True
    use_hanged: bool = True
    max_train_samples: int = -1

    # Training
    output_dir: str = "./flux_lora_out"
    resolution: int = 512  # Start with 512, can increase if memory allows
    center_crop: bool = True
    random_flip: bool = False

    train_batch_size: int = 1
    grad_accum: int = 4
    max_train_steps: int = 1500

    # LoRA
    rank: int = 16
    alpha: int = 16
    init_lora_weights: str = "gaussian"
    transformer_only: bool = True  # Only train transformer LoRA, not text encoders

    # IP-Adapter (image conditioning)
    use_ip_adapter: bool = True
    image_encoder_model: str = "openai/clip-vit-large-patch14"
    ip_scale: float = 0.6
    ip_num_tokens: int = 4  # Number of image tokens to inject

    # Optimizer
    lr: float = 1e-5
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    max_grad_norm: float = 1.0

    # Scheduler
    lr_scheduler: str = "constant_with_warmup"
    warmup_ratio: float = 0.1

    # Other
    mixed_precision: str = "bf16"
    gradient_checkpointing: bool = True
    checkpointing_steps: int = 500
    seed: int = 42
    logging_steps: int = 10
    validation_steps: int = 100


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Train Flux LoRA")
    parser.add_argument("--rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--output_dir", type=str, default="./flux_lora_out", help="Output directory")
    parser.add_argument("--max_train_steps", type=int, default=1500, help="Max training steps")
    parser.add_argument("--resolution", type=int, default=512, help="Training resolution")

    cli_args = parser.parse_args()

    # Create Args dataclass with overrides from CLI
    args = Args()
    if cli_args.rank != 16:
        args.rank = cli_args.rank
        args.alpha = cli_args.rank  # Keep alpha = rank
    if cli_args.output_dir != "./flux_lora_out":
        args.output_dir = cli_args.output_dir
    if cli_args.max_train_steps != 1500:
        args.max_train_steps = cli_args.max_train_steps
    if cli_args.resolution != 512:
        args.resolution = cli_args.resolution

    return args


# ----------------------------
# Dataset
# ----------------------------
def load_paired_list(root: str, use_ground: bool, use_hanged: bool, max_samples: int = -1):
    """
    Load paired images: guidance images (ground/hanged) -> target (straightened)
    Returns list of dicts with 'guidance_paths', 'target_path', 'num'
    """
    root = Path(root)
    ground_dir = root / "on_ground_white_bg"
    hanged_dir = root / "hanged_white_bg"
    target_dir = root / "straightened"

    if not target_dir.exists():
        raise FileNotFoundError(f"Target directory not found: {target_dir}")

    # Get all target images
    target_files = {}
    for ext in ["*.png", "*.jpg", "*.jpeg", "*.webp"]:
        for f in target_dir.glob(ext):
            # Extract number from filename (e.g., straightened_10.png -> 10)
            num_str = ''.join(c for c in f.stem if c.isdigit())
            if num_str:
                target_files[int(num_str)] = f

    items = []
    for num in sorted(target_files.keys()):
        guidance_paths = []

        # Look for matching ground image
        if use_ground and ground_dir.exists():
            for ext in [".png", ".jpg", ".jpeg", ".webp"]:
                ground_file = ground_dir / f"on_ground{num}{ext}"
                if ground_file.exists():
                    guidance_paths.append(str(ground_file))
                    break

        # Look for matching hanged image
        if use_hanged and hanged_dir.exists():
            for ext in [".png", ".jpg", ".jpeg", ".webp"]:
                hanged_file = hanged_dir / f"hanged_{num}{ext}"
                if hanged_file.exists():
                    guidance_paths.append(str(hanged_file))
                    break

        if len(guidance_paths) > 0:
            items.append({
                "guidance_paths": guidance_paths,
                "target_path": str(target_files[num]),
                "num": num,
            })

        if max_samples > 0 and len(items) >= max_samples:
            break

    print(f"[dataset] Loaded {len(items)} paired samples")
    return items


def make_transforms(resolution: int, center_crop: bool, random_flip: bool):
    """Create image transforms"""
    t_list = [transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR)]
    if center_crop:
        t_list.append(transforms.CenterCrop(resolution))
    if random_flip:
        t_list.append(transforms.RandomHorizontalFlip())
    t_list.append(transforms.ToTensor())
    t_list.append(transforms.Normalize([0.5], [0.5]))
    return transforms.Compose(t_list)


# ----------------------------
# Image Projection Layer
# ----------------------------
class ImageProjection(torch.nn.Module):
    """Project CLIP image embeddings to T5 text embedding dimension, expanded to K tokens"""
    def __init__(self, clip_dim: int, text_dim: int, num_tokens: int = 4):
        super().__init__()
        self.num_tokens = num_tokens
        # Project to text_dim * num_tokens, then reshape to [B, num_tokens, text_dim]
        self.proj = torch.nn.Linear(clip_dim, text_dim * num_tokens, bias=False)
        # Initialize with small values for stability
        torch.nn.init.normal_(self.proj.weight, std=0.02)
        self.text_dim = text_dim

    def forward(self, image_embeds):
        """
        Args:
            image_embeds: [B, clip_dim] from CLIP image encoder
        Returns:
            [B, num_tokens, text_dim] projected and expanded to K tokens
        """
        projected = self.proj(image_embeds)  # [B, text_dim * num_tokens]
        return projected.view(-1, self.num_tokens, self.text_dim)  # [B, num_tokens, text_dim]


def encode_image_embeds(image_encoder, image_processor, image_pils, device):
    """
    Encode PIL images with CLIP vision encoder
    Similar to SDXL IP-Adapter approach

    Args:
        image_encoder: CLIPVisionModelWithProjection
        image_processor: CLIPImageProcessor
        image_pils: List[PIL.Image] - guidance images
        device: torch device

    Returns:
        torch.Tensor [B, projection_dim] - CLIP image embeddings
    """
    image_encoder.eval()
    embeds_list = []

    with torch.no_grad():
        for img in image_pils:
            # Preprocess image
            pixel_values = image_processor(images=img, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(device, dtype=torch.bfloat16)

            # Encode
            output = image_encoder(pixel_values)
            img_embed = output.image_embeds  # [1, projection_dim]
            embeds_list.append(img_embed)

    # Concatenate all embeddings
    embeds = torch.cat(embeds_list, dim=0)  # [B, projection_dim]
    return embeds.to(torch.float32)  # Return in fp32 like SDXL


# ----------------------------
# Save
# ----------------------------
def save_lora(pipe: FluxPipeline, transformer, image_proj, args: Args, tag: str):
    """Save LoRA weights and image projection"""
    from safetensors.torch import save_file

    state: Dict[str, Any] = {}
    transformer = Accelerator().unwrap_model(transformer)
    transformer_sd = get_peft_model_state_dict(transformer)

    # Add transformer prefix to all keys
    state.update({f"transformer.{k}": v for k, v in transformer_sd.items()})

    # Add image projection if it exists
    if image_proj is not None:
        image_proj = Accelerator().unwrap_model(image_proj)
        state.update({f"image_proj.{k}": v for k, v in image_proj.state_dict().items()})

    weight_name = f"lora_flux_rank{args.rank}_steps{args.max_train_steps}_{tag}.safetensors"
    save_path = os.path.join(args.output_dir, weight_name)
    save_file(state, save_path)
    print(f"[saved] {save_path}")


# ----------------------------
# Main
# ----------------------------
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=os.path.join(args.output_dir, "logs"))
    accelerator = Accelerator(
        gradient_accumulation_steps=args.grad_accum,
        mixed_precision=args.mixed_precision,
        log_with=None,
        project_config=project_config,
    )
    device = accelerator.device

    # Load Flux pipeline
    print(f"[Loading] Flux pipeline from {args.pretrained_model}")
    pipe = FluxPipeline.from_pretrained(
        args.pretrained_model,
        torch_dtype=torch.bfloat16 if args.mixed_precision == "bf16" else torch.float16,
    )

    vae: AutoencoderKL = pipe.vae
    transformer: FluxTransformer2DModel = pipe.transformer
    text_encoder = pipe.text_encoder
    text_encoder_2 = pipe.text_encoder_2
    tokenizer = pipe.tokenizer
    tokenizer_2 = pipe.tokenizer_2

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    # Freeze everything except LoRA
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    transformer.requires_grad_(False)

    # Set eval mode
    vae.eval()
    text_encoder.eval()
    text_encoder_2.eval()

    # Load CLIP image encoder for IP-Adapter-style conditioning
    image_encoder = None
    image_processor = None
    image_proj = None

    if args.use_ip_adapter:
        print(f"[IP-Adapter] Loading CLIP image encoder: {args.image_encoder_model}")
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            args.image_encoder_model,
            torch_dtype=torch.bfloat16
        )
        image_processor = CLIPImageProcessor.from_pretrained(args.image_encoder_model)
        image_encoder.requires_grad_(False)
        image_encoder.eval()
        image_encoder.to(device)  # Keep image encoder on GPU (frozen)

        # Create image projection layer: CLIP dim -> T5 dim
        clip_projection_dim = image_encoder.config.projection_dim  # 768 for CLIP-L
        t5_hidden_dim = text_encoder_2.config.d_model  # 4096 for T5-XXL

        print(f"[IP-Adapter] Creating projection: {clip_projection_dim} -> {t5_hidden_dim} x {args.ip_num_tokens} tokens")
        image_proj = ImageProjection(clip_projection_dim, t5_hidden_dim, num_tokens=args.ip_num_tokens)
        image_proj.to(device, dtype=torch.float32)  # Keep in fp32 for better precision
        image_proj.train()  # Image projection is trainable

    # Scheduler
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model, subfolder="scheduler"
    )

    # LoRA on Transformer
    print("[LoRA] Applying to transformer")
    transformer_lcfg = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        init_lora_weights=args.init_lora_weights,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    )
    transformer = get_peft_model(transformer, transformer_lcfg)

    # Upcast trainable params to fp32
    for p in transformer.parameters():
        if p.requires_grad:
            p.data = p.data.to(torch.float32)

    # Dataset
    items = load_paired_list(args.data_root, args.use_ground, args.use_hanged, args.max_train_samples)
    if len(items) == 0:
        raise RuntimeError(f"No paired items found under {args.data_root}")

    captions = ["professional product photo, garment laid flat on white background, straightened and dewrinkled, studio catalog, centered, evenly lit"] * len(items)

    ds = Dataset.from_dict({
        "guidance_paths": [x["guidance_paths"] for x in items],
        "target_path": [x["target_path"] for x in items],
        "caption": captions,
        "num": [x["num"] for x in items],
    })

    tx = make_transforms(args.resolution, args.center_crop, args.random_flip)

    def preprocess(examples):
        timgs = [Image.open(p).convert("RGB") for p in examples["target_path"]]
        examples["pixel_values"] = [tx(im) for im in timgs]

        # Tokenize with both encoders
        ids_1 = tokenizer(
            examples["caption"],
            max_length=tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids

        ids_2 = tokenizer_2(
            examples["caption"],
            max_length=tokenizer_2.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids

        examples["input_ids_1"] = ids_1
        examples["input_ids_2"] = ids_2
        return examples

    with accelerator.main_process_first():
        ds = ds.with_transform(preprocess)

    def collate(batch):
        pixel_values = torch.stack([b["pixel_values"] for b in batch]).to(memory_format=torch.contiguous_format).float()
        input_ids_1 = torch.stack([b["input_ids_1"] for b in batch])
        input_ids_2 = torch.stack([b["input_ids_2"] for b in batch])
        guidance_paths = [b["guidance_paths"] for b in batch]
        nums = [b["num"] for b in batch]
        return {
            "pixel_values": pixel_values,
            "input_ids_1": input_ids_1,
            "input_ids_2": input_ids_2,
            "guidance_paths": guidance_paths,
            "nums": nums,
        }

    train_loader = DataLoader(
        ds, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate, num_workers=0
    )

    # Optimizer - split weight decay for LoRA and image projection
    lora_params = [p for p in transformer.parameters() if p.requires_grad]
    proj_params = [p for p in image_proj.parameters()] if image_proj is not None else []

    param_groups = [
        {"params": lora_params, "weight_decay": 0.0},  # No weight decay for LoRA
        {"params": proj_params, "weight_decay": 1e-4},  # Small weight decay for image_proj
    ]

    optimizer = torch.optim.AdamW(
        param_groups,
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_eps,
    )

    # LR scheduler
    from transformers import get_scheduler
    num_training_steps = args.max_train_steps
    num_warmup_steps = int(args.warmup_ratio * num_training_steps)
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps * args.grad_accum,
        num_training_steps=num_training_steps * args.grad_accum,
    )

    # Prepare with accelerator
    if image_proj is not None:
        transformer, image_proj, optimizer, train_loader, lr_scheduler = accelerator.prepare(
            transformer, image_proj, optimizer, train_loader, lr_scheduler
        )
    else:
        transformer, optimizer, train_loader, lr_scheduler = accelerator.prepare(
            transformer, optimizer, train_loader, lr_scheduler
        )

    # Move encoders to device - keep text encoders on CPU, only VAE on GPU
    text_encoder.to("cpu").eval()
    text_encoder_2.to("cpu").eval()
    vae.to(device)  # VAE stays on GPU

    # Training loop
    print(f"\n[Training] Starting {args.max_train_steps} steps...")
    print(f"  Batch size = {args.train_batch_size}")
    print(f"  Gradient accumulation = {args.grad_accum}")
    print(f"  LoRA rank = {args.rank}")
    print(f"  Learning rate = {args.lr}\n")

    global_step = 0
    progress_bar = tqdm(total=args.max_train_steps, disable=not accelerator.is_local_main_process)

    while global_step < args.max_train_steps:
        for batch in train_loader:
            with accelerator.accumulate(transformer):
                # Encode text on CPU, move only embeddings to GPU
                with torch.no_grad():
                    # Text encoders stay on CPU, input_ids must be on CPU too
                    input_ids_1_cpu = batch["input_ids_1"].cpu()
                    input_ids_2_cpu = batch["input_ids_2"].cpu()

                    # CLIP encoder (text_encoder) - has pooled output
                    enc_1 = text_encoder(input_ids_1_cpu, output_hidden_states=False)
                    prompt_embeds_1_cpu = enc_1.last_hidden_state
                    pooled_embeds_cpu = enc_1.pooler_output  # CLIP has pooler_output

                    # T5 encoder (text_encoder_2) - no pooled output
                    enc_2 = text_encoder_2(input_ids_2_cpu, output_hidden_states=False)
                    prompt_embeds_2_cpu = enc_2.last_hidden_state

                    # Move only embeddings to GPU
                    # Flux uses T5 as main encoder_hidden_states, CLIP pooled as pooled_projections
                    prompt_embeds = prompt_embeds_2_cpu.to(device, dtype=torch.bfloat16, non_blocking=True)  # T5
                    pooled_embeds = pooled_embeds_cpu.to(device, dtype=torch.bfloat16, non_blocking=True)    # CLIP pooled

                # Encode guidance images (if using IP-Adapter)
                if args.use_ip_adapter and image_encoder is not None:
                    # Load guidance images (like SDXL IP-Adapter)
                    guidance_images_batch = []
                    for paths in batch["guidance_paths"]:
                        # Load all guidance images for this sample (e.g., ground + hanged)
                        imgs = [Image.open(p).convert("RGB") for p in paths]
                        guidance_images_batch.append(imgs)

                    # Encode each sample's guidance images
                    image_embeds_list = []
                    with torch.no_grad():
                        for imgs in guidance_images_batch:
                            # Encode all guidance images and average (like SDXL)
                            img_embeds = encode_image_embeds(image_encoder, image_processor, imgs, device)
                            # Average multiple guidance images (ground + hanged)
                            avg_embed = img_embeds.mean(dim=0, keepdim=True)  # [1, clip_dim]
                            image_embeds_list.append(avg_embed)

                        # Stack into batch
                        image_embeds = torch.cat(image_embeds_list, dim=0)  # [B, clip_dim]

                    # Project to T5 dimension and expand to K tokens
                    # image_proj expects fp32 input for better precision
                    projected_image_embeds = image_proj(image_embeds.to(device, dtype=torch.float32))  # [B, num_tokens, t5_dim]

                    # Scale like IP-Adapter
                    projected_image_embeds = projected_image_embeds * args.ip_scale

                    # Concatenate image tokens with text tokens
                    # Image becomes additional "text" tokens: [B, num_tokens, t5_dim]
                    # Cast to bf16 to avoid upcasting the entire prompt sequence to fp32
                    image_tokens = projected_image_embeds.to(torch.bfloat16)  # [B, num_tokens, t5_dim]

                    # Prepend to text embeddings: [B, seq_len, dim] -> [B, num_tokens+seq_len, dim]
                    prompt_embeds = torch.cat([image_tokens, prompt_embeds], dim=1)

                # Encode target images to latents
                pixel_values = batch["pixel_values"].to(device, dtype=vae.dtype)
                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = (latents - vae.config.shift_factor) * vae.config.scaling_factor

                # Add noise (flow matching)
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]

                # Random timesteps
                # Flux uses continuous time from 0 to 1
                timesteps = torch.rand(bsz, device=device)

                # Flow matching: interpolate between noise and data
                noisy_latents = (1 - timesteps.view(-1, 1, 1, 1)) * latents + timesteps.view(-1, 1, 1, 1) * noise

                # Pack latents into patches for Flux transformer
                # Flux uses 2x2 patches: (B, C, H, W) -> (B, H/2 * W/2, C*4)
                b, c, h, w = noisy_latents.shape
                patch_size = 2

                # Reshape to patches: (B, C, H, W) -> (B, C, H/2, 2, W/2, 2) -> (B, H/2, W/2, C, 2, 2) -> (B, H/2*W/2, C*4)
                noisy_latents_packed = noisy_latents.reshape(b, c, h // patch_size, patch_size, w // patch_size, patch_size)
                noisy_latents_packed = noisy_latents_packed.permute(0, 2, 4, 1, 3, 5).reshape(b, (h // patch_size) * (w // patch_size), c * patch_size * patch_size)

                # Pack target too for loss calculation
                latents_packed = latents.reshape(b, c, h // patch_size, patch_size, w // patch_size, patch_size)
                latents_packed = latents_packed.permute(0, 2, 4, 1, 3, 5).reshape(b, (h // patch_size) * (w // patch_size), c * patch_size * patch_size)

                noise_packed = noise.reshape(b, c, h // patch_size, patch_size, w // patch_size, patch_size)
                noise_packed = noise_packed.permute(0, 2, 4, 1, 3, 5).reshape(b, (h // patch_size) * (w // patch_size), c * patch_size * patch_size)

                # Prepare positional IDs for Flux (2D tensors, no batch dimension)
                num_patches = (h // patch_size) * (w // patch_size)
                img_ids = torch.zeros(num_patches, 3, device=device, dtype=torch.bfloat16)
                # txt_ids needs to match the sequence length (which now includes image token if IP-Adapter is used)
                txt_ids = torch.zeros(prompt_embeds.shape[1], 3, device=device, dtype=torch.bfloat16)

                # Explicit dtype casts to avoid silent upcasts
                noisy_latents_packed = noisy_latents_packed.to(torch.bfloat16)
                prompt_embeds = prompt_embeds.to(torch.bfloat16)
                pooled_embeds = pooled_embeds.to(torch.bfloat16)
                txt_ids = txt_ids.to(torch.bfloat16)
                img_ids = img_ids.to(torch.bfloat16)

                # Predict velocity
                model_pred = transformer(
                    hidden_states=noisy_latents_packed,
                    timestep=timesteps,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_embeds,
                    txt_ids=txt_ids,
                    img_ids=img_ids,
                    return_dict=False,
                )[0]

                # Flow matching loss: predict the velocity (noise - data)
                target = noise_packed - latents_packed

                # MSE loss
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                # Backward
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    params_to_clip = [p for p in transformer.parameters() if p.requires_grad]
                    if image_proj is not None:
                        params_to_clip.extend([p for p in image_proj.parameters() if p.requires_grad])
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                progress_bar.update(1)

                if global_step % args.logging_steps == 0:
                    logs = {
                        "loss": loss.detach().item(),
                        "lr": lr_scheduler.get_last_lr()[0],
                        "step": global_step,
                    }
                    progress_bar.set_postfix(**logs)

                if global_step % args.checkpointing_steps == 0:
                    save_lora(pipe, transformer, image_proj, args, f"step{global_step}")

                if global_step >= args.max_train_steps:
                    break

    # Final save
    save_lora(pipe, transformer, image_proj, args, f"step{global_step}")
    accelerator.end_training()
    print("\n[Done] Training complete!")


if __name__ == "__main__":
    main()
