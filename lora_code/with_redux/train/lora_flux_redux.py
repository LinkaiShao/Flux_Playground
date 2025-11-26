#!/usr/bin/env python3
"""
Train LoRA on FLUX with Redux image conditioning.
Uses pretrained FLUX.1-Redux for image encoding - only trains LoRA weights.
"""

import os
import warnings
from dataclasses import dataclass
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
    FluxPriorReduxPipeline,
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
)
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

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
    redux_model: str = "/home/link/Desktop/Code/fashion gen testing/flux-redux"

    # Data
    data_root: str = "/home/link/Desktop/Code/fashion gen testing/straighten_image_data"
    use_ground: bool = True
    use_hanged: bool = True
    max_train_samples: int = -1

    # Training
    output_dir: str = "./flux_lora_out"
    resolution: int = 512
    center_crop: bool = True
    random_flip: bool = False

    train_batch_size: int = 1
    grad_accum: int = 4
    max_train_steps: int = 1500

    # LoRA
    rank: int = 16
    alpha: int = 16
    init_lora_weights: str = "gaussian"
    transformer_only: bool = True

    # Optimizer
    lr: float = 1e-5
    weight_decay: float = 0.0  # No weight decay for LoRA
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
    parser = argparse.ArgumentParser(description="Train Flux LoRA with Redux")
    parser.add_argument("--rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--output_dir", type=str, default="./flux_lora_out", help="Output directory")
    parser.add_argument("--max_train_steps", type=int, default=1500, help="Max training steps")
    parser.add_argument("--resolution", type=int, default=512, help="Training resolution")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--checkpointing_steps", type=int, default=500, help="Save checkpoint every N steps")

    args_dict = vars(parser.parse_args())
    base_args = Args()
    for k, v in args_dict.items():
        if hasattr(base_args, k):
            setattr(base_args, k, v)
    return base_args


def load_dataset(args: Args):
    """Load image triplets: guidance, target, caption"""
    import re
    guidance_images = []
    target_images = []
    captions = []

    # Load original straighten_image_data dataset
    target_dir = Path(args.data_root) / "straightened"
    target_files = sorted(target_dir.glob("*.png"))

    for target_path in target_files:
        stem = target_path.stem

        # Extract number from stem (e.g., "straightened_10" -> "10")
        match = re.search(r'(\d+)$', stem)
        if not match:
            continue
        number = match.group(1)

        # Find matching guidance image
        guidance_path = None
        if args.use_ground:
            ground_path = Path(args.data_root) / "on_ground_white_bg" / f"on_ground{number}.jpg"
            if ground_path.exists():
                guidance_path = ground_path

        if guidance_path is None and args.use_hanged:
            hanged_path = Path(args.data_root) / "hanged_white_bg" / f"hanged{number}.jpg"
            if hanged_path.exists():
                guidance_path = hanged_path

        if guidance_path is not None:
            guidance_images.append(str(guidance_path))
            target_images.append(str(target_path))
            captions.append("professional product photo, garment laid flat on white background, straightened and dewrinkled, studio catalog, centered, evenly lit, sharp edges")

    # Load grailed_crawl dataset (biref_processed -> gpt_processed)
    grailed_root = Path(args.data_root).parent / "grailed_crawl"
    biref_dir = grailed_root / "biref processed"
    gpt_dir = grailed_root / "gpt_processed"

    if biref_dir.exists() and gpt_dir.exists():
        biref_files = sorted(biref_dir.glob("white_bg_grailed_*.jpg"))

        for biref_path in biref_files:
            # Extract number (e.g., "white_bg_grailed_0001.jpg" -> "0001")
            match = re.search(r'white_bg_grailed_(\d{4})\.jpg', biref_path.name)
            if not match:
                continue
            number = match.group(1)

            # Check if matching gpt_processed file exists
            gpt_path = gpt_dir / f"gpt_grailed_{number}.png"
            if gpt_path.exists():
                guidance_images.append(str(biref_path))
                target_images.append(str(gpt_path))
                captions.append("professional product photo, garment laid flat on white background, straightened and dewrinkled, studio catalog, centered, evenly lit, sharp edges")

    dataset_dict = {
        "guidance_image": guidance_images,
        "target_image": target_images,
        "caption": captions
    }

    dataset = Dataset.from_dict(dataset_dict)

    print(f"Loaded dataset: {len(dataset)} total samples")
    if args.max_train_samples > 0:
        dataset = dataset.select(range(min(args.max_train_samples, len(dataset))))

    return dataset


def collate_fn(examples, image_transforms, vae, redux_pipe, device, weight_dtype):
    """Preprocess batch and encode with Redux"""
    guidance_images = [Image.open(ex["guidance_image"]).convert("RGB") for ex in examples]
    target_images = [Image.open(ex["target_image"]).convert("RGB") for ex in examples]
    text_embeds_t5 = torch.stack([torch.tensor(ex["text_embeds_t5"], dtype=torch.float32) for ex in examples])
    text_pooled_embeds = torch.stack([torch.tensor(ex["text_pooled_embeds"], dtype=torch.float32) for ex in examples])

    # Transform images
    guidance_pil = [image_transforms(img) for img in guidance_images]
    target_pil = [image_transforms(img) for img in target_images]

    # Encode target images to latents (temporarily move VAE to GPU)
    target_tensors = torch.stack([transforms.ToTensor()(img) for img in target_pil])
    target_tensors = target_tensors * 2.0 - 1.0

    with torch.no_grad():
        vae.to(device)
        target_tensors = target_tensors.to(device, dtype=weight_dtype)
        latents = vae.encode(target_tensors).latent_dist.sample()
        latents = (latents - vae.config.shift_factor) * vae.config.scaling_factor
        latents = latents.to(device).clone()  # Keep on device and clone to free VAE memory
        vae.to("cpu")  # Move back to CPU
        torch.cuda.empty_cache()

    # Encode guidance images with Redux (image only)
    with torch.no_grad():
        redux_outputs = []
        for guid_img in guidance_pil:
            redux_out = redux_pipe(image=guid_img)
            # Redux outputs image embeddings in the text embedding space
            redux_outputs.append({
                "prompt_embeds": redux_out.prompt_embeds.to(device).clone(),
                "pooled_prompt_embeds": redux_out.pooled_prompt_embeds.to(device).clone()
            })

    # Stack Redux embeddings
    redux_prompt_embeds = torch.cat([out["prompt_embeds"] for out in redux_outputs], dim=0)
    redux_pooled_embeds = torch.cat([out["pooled_prompt_embeds"] for out in redux_outputs], dim=0)

    # Combine text embeddings and Redux image embeddings (concatenate along sequence dimension)
    prompt_embeds = torch.cat([text_embeds_t5.to(device), redux_prompt_embeds], dim=1)
    pooled_prompt_embeds = text_pooled_embeds.to(device) + redux_pooled_embeds

    return {
        "latents": latents,
        "prompt_embeds": prompt_embeds,
        "pooled_prompt_embeds": pooled_prompt_embeds,
    }


def pack_latents(latents, batch_size, num_channels_latents, height, width):
    """Pack latents for Flux (2x2 patchify)"""
    latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)
    return latents


def unpack_latents(latents, height, width, vae_scale_factor):
    """Unpack latents from Flux format"""
    batch_size, num_patches, channels = latents.shape

    height = height // vae_scale_factor
    width = width // vae_scale_factor

    latents = latents.view(batch_size, height, width, channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)

    latents = latents.reshape(batch_size, channels // (2 * 2), height * 2, width * 2)

    return latents


def main():
    args = parse_args()

    # Setup accelerator
    project_config = ProjectConfiguration(
        project_dir=args.output_dir,
        logging_dir=os.path.join(args.output_dir, "logs"),
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.grad_accum,
        mixed_precision=args.mixed_precision,
        project_config=project_config,
    )

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # Set seed
    torch.manual_seed(args.seed)

    # Load models
    print("Loading models...")
    weight_dtype = torch.bfloat16

    # Load Redux pipeline for image encoding (keep on CPU to save GPU memory)
    print("Loading Redux pipeline...")
    redux_pipe = FluxPriorReduxPipeline.from_pretrained(
        args.redux_model,
        torch_dtype=weight_dtype
    )
    redux_pipe.to("cpu")  # Keep on CPU to save GPU memory

    # Disable gradients for Redux components
    for component in [redux_pipe.image_encoder, redux_pipe.image_embedder, redux_pipe.feature_extractor]:
        if hasattr(component, 'requires_grad_'):
            component.requires_grad_(False)
        if hasattr(component, 'eval'):
            component.eval()

    # Load VAE (keep on CPU to save GPU memory during data loading)
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model,
        subfolder="vae",
        torch_dtype=weight_dtype
    )
    vae.to("cpu")
    vae.requires_grad_(False)
    vae.eval()

    # Load text encoders for encoding text prompts (move to CPU to save memory)
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model, subfolder="tokenizer")
    tokenizer_2 = T5TokenizerFast.from_pretrained(args.pretrained_model, subfolder="tokenizer_2")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model, subfolder="text_encoder", torch_dtype=weight_dtype)
    text_encoder_2 = T5EncoderModel.from_pretrained(args.pretrained_model, subfolder="text_encoder_2", torch_dtype=weight_dtype)
    text_encoder.to("cpu")
    text_encoder_2.to("cpu")
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    text_encoder.eval()
    text_encoder_2.eval()

    # Load transformer and apply LoRA
    print(f"Loading transformer and applying LoRA (rank={args.rank})...")
    transformer = FluxTransformer2DModel.from_pretrained(
        args.pretrained_model,
        subfolder="transformer",
        torch_dtype=weight_dtype
    )

    # Configure LoRA
    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        init_lora_weights=args.init_lora_weights,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )

    transformer = get_peft_model(transformer, lora_config)
    transformer.print_trainable_parameters()

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    # Noise scheduler
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model,
        subfolder="scheduler"
    )

    # Optimizer (only LoRA parameters)
    optimizer = torch.optim.AdamW(
        transformer.parameters(),
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
    )

    # Dataset
    print("Loading dataset...")
    dataset = load_dataset(args)
    print(f"Loaded {len(dataset)} samples")

    # Pre-encode text prompt once (all samples use same caption)
    print("Pre-encoding text prompt...")
    text_encoder.to(accelerator.device)
    text_encoder_2.to(accelerator.device)

    caption = "professional product photo, garment laid flat on white background, straightened and dewrinkled, studio catalog, centered, evenly lit, sharp edges"

    with torch.no_grad():
        # Tokenize
        text_inputs = tokenizer([caption], padding="max_length", max_length=77, truncation=True, return_tensors="pt")
        text_inputs_2 = tokenizer_2([caption], padding="max_length", max_length=512, truncation=True, return_tensors="pt")

        # Encode with CLIP
        prompt_embeds_clip = text_encoder(text_inputs.input_ids.to(accelerator.device), output_hidden_states=False)
        text_pooled_embeds = prompt_embeds_clip.pooler_output[0]  # [768]

        # Encode with T5
        text_embeds_t5 = text_encoder_2(text_inputs_2.input_ids.to(accelerator.device), output_hidden_states=False)[0][0]  # [seq_len, 4096]

        # Move to CPU
        text_embeds_t5 = text_embeds_t5.cpu()
        text_pooled_embeds = text_pooled_embeds.cpu()

    print(f"Text prompt encoded (will be reused for all {len(dataset)} samples)")

    # Free text encoders
    del text_encoder, text_encoder_2
    torch.cuda.empty_cache()
    print("Text encoders freed")

    # Image transforms
    image_transforms = transforms.Compose([
        transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(args.resolution) if args.center_crop else transforms.Lambda(lambda x: x),
        transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
    ])

    # Pre-encode all images with VAE
    print("Pre-encoding all images with VAE...")
    vae.to(accelerator.device)

    all_latents = []
    with torch.no_grad():
        for example in tqdm(dataset, desc="Encoding images"):
            target_image = Image.open(example["target_image"]).convert("RGB")
            target_pil = image_transforms(target_image)
            target_tensor = transforms.ToTensor()(target_pil)
            target_tensor = target_tensor * 2.0 - 1.0
            target_tensor = target_tensor.unsqueeze(0).to(accelerator.device, dtype=weight_dtype)

            latent = vae.encode(target_tensor).latent_dist.sample()
            latent = (latent - vae.config.shift_factor) * vae.config.scaling_factor
            all_latents.append(latent.cpu())

    # Free VAE
    del vae
    torch.cuda.empty_cache()
    print("VAE freed")

    # Pre-encode all guidance images with Redux
    print("Pre-encoding all guidance images with Redux...")
    redux_pipe.to(accelerator.device)

    all_redux_prompt_embeds = []
    all_redux_pooled_embeds = []
    with torch.no_grad():
        for example in tqdm(dataset, desc="Encoding guidance images"):
            guidance_image = Image.open(example["guidance_image"]).convert("RGB")
            guidance_pil = image_transforms(guidance_image)

            redux_out = redux_pipe(image=guidance_pil)
            all_redux_prompt_embeds.append(redux_out.prompt_embeds.cpu())
            all_redux_pooled_embeds.append(redux_out.pooled_prompt_embeds.cpu())

    # Free Redux
    del redux_pipe
    torch.cuda.empty_cache()
    print("Redux freed")

    # Combine text and redux embeddings
    print("Combining text and redux embeddings...")
    combined_prompt_embeds = []
    combined_pooled_embeds = []

    for i in range(len(dataset)):
        # Combine text embeddings and Redux image embeddings
        prompt_embeds = torch.cat([text_embeds_t5.unsqueeze(0), all_redux_prompt_embeds[i]], dim=1)
        pooled_embeds = text_pooled_embeds + all_redux_pooled_embeds[i].squeeze(0)

        combined_prompt_embeds.append(prompt_embeds.squeeze(0))
        combined_pooled_embeds.append(pooled_embeds)

    # Create simple collate function for pre-encoded data
    def simple_collate_fn(indices):
        latents = torch.stack([all_latents[i].squeeze(0) for i in indices])
        prompt_embeds = torch.stack([combined_prompt_embeds[i] for i in indices])
        pooled_embeds = torch.stack([combined_pooled_embeds[i] for i in indices])

        return {
            "latents": latents,
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled_embeds,
        }

    # Create index dataset
    from torch.utils.data import TensorDataset
    indices_tensor = torch.arange(len(dataset))
    index_dataset = TensorDataset(indices_tensor)

    # DataLoader
    train_dataloader = DataLoader(
        index_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=lambda batch: simple_collate_fn([b[0].item() for b in batch]),
        num_workers=0,
    )

    # LR scheduler
    num_training_steps = args.max_train_steps
    num_warmup_steps = int(args.warmup_ratio * num_training_steps)

    from torch.optim.lr_scheduler import LambdaLR
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return 1.0

    lr_scheduler = LambdaLR(optimizer, lr_lambda)

    # Prepare with accelerator
    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )

    # Training loop
    print("Starting training...")
    global_step = 0
    progress_bar = tqdm(total=args.max_train_steps, disable=not accelerator.is_local_main_process)

    transformer.train()

    for epoch in range(999):  # Large number, will break when max_steps reached
        for batch in train_dataloader:
            with accelerator.accumulate(transformer):
                latents = batch["latents"]
                prompt_embeds = batch["prompt_embeds"]
                pooled_prompt_embeds = batch["pooled_prompt_embeds"]

                bsz = latents.shape[0]

                # Sample noise
                noise = torch.randn_like(latents)

                # Sample timesteps (0 to 1 for flow matching)
                u = torch.rand(bsz, device=latents.device)
                indices = (u * noise_scheduler.config.num_train_timesteps).long()
                timesteps = noise_scheduler.timesteps[indices.cpu()].to(latents.device)

                # Interpolate between noise and data
                sigmas = u.view(-1, 1, 1)
                noisy_latents = (1 - sigmas) * latents + sigmas * noise

                # Pack latents for Flux
                latent_h, latent_w = latents.shape[2], latents.shape[3]

                latents_packed = pack_latents(
                    latents,
                    bsz,
                    latents.shape[1],
                    latent_h,
                    latent_w
                )

                noise_packed = pack_latents(
                    noise,
                    bsz,
                    noise.shape[1],
                    latent_h,
                    latent_w
                )

                noisy_latents_packed = pack_latents(
                    noisy_latents,
                    bsz,
                    noisy_latents.shape[1],
                    latent_h,
                    latent_w
                )

                # Prepare positional IDs for Flux
                num_patches = latent_h * latent_w // 4  # After 2x2 packing
                img_ids = torch.zeros(num_patches, 3, device=latents.device, dtype=weight_dtype)
                txt_ids = torch.zeros(prompt_embeds.shape[1], 3, device=latents.device, dtype=weight_dtype)

                # Predict velocity
                model_pred = transformer(
                    hidden_states=noisy_latents_packed.to(weight_dtype),
                    timestep=timesteps.to(weight_dtype) / 1000,
                    encoder_hidden_states=prompt_embeds.to(weight_dtype),
                    pooled_projections=pooled_prompt_embeds.to(weight_dtype),
                    txt_ids=txt_ids,
                    img_ids=img_ids,
                    return_dict=False,
                )[0]

                model_pred = model_pred.to(torch.float32)

                # Flow matching loss: predict velocity (noise - data)
                target = (noise_packed - latents_packed).to(torch.float32)
                loss = F.mse_loss(model_pred, target, reduction="mean")

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(transformer.parameters(), args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if global_step % args.logging_steps == 0:
                    logs = {
                        "loss": loss.detach().item(),
                        "lr": lr_scheduler.get_last_lr()[0]
                    }
                    progress_bar.set_postfix(**logs)

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(
                            args.output_dir,
                            f"lora_flux_rank{args.rank}_steps{args.max_train_steps}_step{global_step}.safetensors"
                        )
                        lora_state_dict = get_peft_model_state_dict(transformer)

                        from safetensors.torch import save_file
                        save_file(lora_state_dict, save_path)
                        print(f"Saved checkpoint: {save_path}")

            if global_step >= args.max_train_steps:
                break

        if global_step >= args.max_train_steps:
            break

    # Save final checkpoint
    if accelerator.is_main_process:
        save_path = os.path.join(
            args.output_dir,
            f"lora_flux_rank{args.rank}_steps{args.max_train_steps}_step{global_step}.safetensors"
        )
        lora_state_dict = get_peft_model_state_dict(transformer)

        from safetensors.torch import save_file
        save_file(lora_state_dict, save_path)
        print(f"Training complete! Final checkpoint: {save_path}")

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()
