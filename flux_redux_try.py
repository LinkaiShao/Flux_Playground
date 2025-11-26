# flux_img2img.py
import os
from pathlib import Path
from PIL import Image
import torch
from diffusers import FluxImg2ImgPipeline  # <-- img2img class

# memory-friendly settings
torch_dtype = torch.float16
model_path  = "./flux1-schnell"  # your local snapshot

# load
pipe = FluxImg2ImgPipeline.from_pretrained(
    model_path, torch_dtype=torch_dtype, local_files_only=True
)
pipe.enable_attention_slicing()
pipe.enable_vae_slicing()
pipe.enable_vae_tiling()
try:
    pipe.enable_sequential_cpu_offload()   # needs accelerate
except Exception:
    pipe.to("cuda")

# image - WITH EXTENSIVE DEBUGGING
inp = Path("u2net_output/birefnet_output_2.png").resolve()
print(f"DEBUG: Loading from: {inp}")
print(f"DEBUG: File exists: {inp.exists()}")
print(f"DEBUG: File size: {inp.stat().st_size}")

# Load original
img_original = Image.open(inp)
print(f"DEBUG: Original mode: {img_original.mode}, size: {img_original.size}")

# Save debug: original as loaded
Path("debug_output").mkdir(exist_ok=True)
img_original.save("debug_output/01_loaded_original.png")
print("DEBUG: Saved debug_output/01_loaded_original.png")

# Convert to RGB - PROPERLY composite onto white background
# (Direct convert() would reveal hidden background in RGB channels!)
if img_original.mode in ("RGBA", "LA"):
    print("DEBUG: Image has alpha channel - compositing onto white background")
    # Create white background
    white_bg = Image.new("RGB", img_original.size, (255, 255, 255))
    # Paste using alpha channel as mask (only pastes non-transparent parts)
    white_bg.paste(img_original, mask=img_original.split()[-1])
    img = white_bg
    print("DEBUG: Composited onto white - hidden background removed")
else:
    img = img_original.convert("RGB")
    print("DEBUG: No alpha channel - simple convert")

print(f"DEBUG: After RGB convert: mode={img.mode}, size={img.size}")
img.save("debug_output/02_after_rgb_convert.png")
print("DEBUG: Saved debug_output/02_after_rgb_convert.png")

# Resize
img = img.resize((896, 896), Image.LANCZOS)
print(f"DEBUG: After resize: mode={img.mode}, size={img.size}")
img.save("debug_output/03_after_resize.png")
print("DEBUG: Saved debug_output/03_after_resize.png")

# run - HIGH strength to prevent background hallucination
prompt = "professional product photography, flat lay garment, pure white seamless background, no environment, no scene, studio lighting, clean and minimal"
negative_prompt = "background texture, environment, room, floor, wall, shadow, depth, person, hanger, mannequin, model, human, wrinkles, heavy shadows, watermark"

print("\nDEBUG: Running FLUX inference...")
print(f"DEBUG: Input image stats - mode={img.mode}, size={img.size}")

out = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image=img,
    strength=0.80,                # HIGH strength = full regeneration, prevents background hallucination
    guidance_scale=6.0,           # Higher guidance for stronger prompt adherence
    num_inference_steps=24,
).images[0]

print(f"DEBUG: Output image stats - mode={out.mode}, size={out.size}")

# Save both input and output for comparison
img.save("debug_output/04_input_to_flux.png")
out.save("debug_output/05_output_from_flux.png")
print("DEBUG: Saved debug_output/04_input_to_flux.png")
print("DEBUG: Saved debug_output/05_output_from_flux.png")

Path("schnell_flux_redux_output").mkdir(parents=True, exist_ok=True)
out.save("schnell_flux_redux_output/flux_schnell_img2img.png")
print("\nFINAL: saved to schnell_flux_redux_output/flux_schnell_img2img.png")
print("\n===== CHECK debug_output/ folder for step-by-step images =====")
