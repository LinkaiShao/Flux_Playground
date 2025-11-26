from diffusers import FlowMatchEulerDiscreteScheduler

scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
    "/home/link/Desktop/Code/fashion gen testing/flux1-schnell",
    subfolder="scheduler"
)

print(f"num_train_timesteps: {scheduler.config.num_train_timesteps}")
print(f"timesteps shape: {scheduler.timesteps.shape}")
print(f"sigmas shape: {scheduler.sigmas.shape}")
print(f"\nFirst 10 timesteps: {scheduler.timesteps[:10]}")
print(f"Last 10 timesteps: {scheduler.timesteps[-10:]}")
print(f"\nFirst 10 sigmas: {scheduler.sigmas[:10]}")
print(f"Last 10 sigmas: {scheduler.sigmas[-10:]}")
print(f"\nSigma range: [{scheduler.sigmas.min()}, {scheduler.sigmas.max()}]")