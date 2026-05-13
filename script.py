import os
import torch
from diffusers import AutoencoderKL
from PIL import Image
import numpy as np

os.environ["HF_HOME"] = "/equilibrium/ldelbene/cache/hf"

DEVICE = "cuda"
IMG_CLEAN_PATH = "output/SD_Inpainting/full_dataset/VAE_noise_mask_MSE/img_2/original_image.png"
IMG_ADV_PATH   = "output/SD_Inpainting/full_dataset/VAE_noise_mask_MSE/img_2/immunized_image.png"



# =====================================================
# LOAD VAE
# =====================================================

vae = AutoencoderKL.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    subfolder="vae",
    torch_dtype=torch.float16,
    local_files_only=True,
).to(DEVICE)
vae.eval()
vae.requires_grad_(False)

with torch.no_grad():
    target = Image.open("target.png").convert("RGB").resize((512, 512))
    target = torch.tensor(np.array(target)).float() / 255.0
    target = target.permute(2, 0, 1).unsqueeze(0)
    target = target * 2.0 - 1.0
    target = target.to(DEVICE, dtype=torch.float16)

    z_gray = vae.encode(target).latent_dist.mean
    print("z_gray mean:", z_gray.mean().item())
    print("z_gray std: ", z_gray.std().item())
    print("z_gray per channel:", [z_gray[0,c].mean().item() for c in range(4)])

# =====================================================
# PREPROCESS
# =====================================================

def load_img(path):
    img = Image.open(path).convert("RGB").resize((512, 512))
    img = torch.tensor(np.array(img)).float() / 255.0
    img = img.permute(2, 0, 1).unsqueeze(0)
    img = img * 2.0 - 1.0
    return img.to(DEVICE, dtype=torch.float16)

img_clean = load_img(IMG_CLEAN_PATH)
img_adv   = load_img(IMG_ADV_PATH)

# =====================================================
# ENCODE
# =====================================================

with torch.no_grad():
    post_clean = vae.encode(img_clean).latent_dist
    post_adv   = vae.encode(img_adv).latent_dist

    z_clean = post_clean.mean
    z_adv   = post_adv.mean

# =====================================================
# STATS
# =====================================================

print("=" * 50)
print("LATENT STATISTICS")
print("=" * 50)

print(f"\n{'':30s} {'CLEAN':>12} {'ADV':>12} {'DIFF':>12}")
print("-" * 70)
print(f"{'mean':30s} {z_clean.mean().item():12.6f} {z_adv.mean().item():12.6f} {(z_clean.mean() - z_adv.mean()).abs().item():12.6f}")
print(f"{'std':30s} {z_clean.std().item():12.6f} {z_adv.std().item():12.6f} {(z_clean.std() - z_adv.std()).abs().item():12.6f}")
print(f"{'min':30s} {z_clean.min().item():12.6f} {z_adv.min().item():12.6f} {(z_clean.min() - z_adv.min()).abs().item():12.6f}")
print(f"{'max':30s} {z_clean.max().item():12.6f} {z_adv.max().item():12.6f} {(z_clean.max() - z_adv.max()).abs().item():12.6f}")
print(f"{'mean abs diff (z)':30s} {'':12} {'':12} {(z_clean - z_adv).abs().mean().item():12.6f}")

print("\n--- Posterior (μ, σ) ---")
print(f"{'mu  clean mean/std':30s} {post_clean.mean.mean().item():12.6f} / {post_clean.mean.std().item():12.6f}")
print(f"{'mu  adv   mean/std':30s} {post_adv.mean.mean().item():12.6f} / {post_adv.mean.std().item():12.6f}")
print(f"{'std clean mean/std':30s} {post_clean.std.mean().item():12.6f} / {post_clean.std.std().item():12.6f}")
print(f"{'std adv   mean/std':30s} {post_adv.std.mean().item():12.6f} / {post_adv.std.std().item():12.6f}")

print("\n--- Per-channel stats ---")
print(f"{'ch':>4} {'z_clean mean':>14} {'z_adv mean':>12} {'z_clean std':>12} {'z_adv std':>12} {'diff':>10}")
print("-" * 70)
for c in range(z_clean.shape[1]):
    print(f"{c:4d} "
          f"{z_clean[0, c].mean().item():14.6f} "
          f"{z_adv[0, c].mean().item():12.6f} "
          f"{z_clean[0, c].std().item():12.6f} "
          f"{z_adv[0, c].std().item():12.6f} "
          f"{(z_clean[0, c] - z_adv[0, c]).abs().mean().item():10.6f}")

# =====================================================
# DECODE E SALVA per ispezione visiva
# =====================================================

with torch.no_grad():
    recon_clean = vae.decode(z_clean * vae.config.scaling_factor).sample
    recon_adv   = vae.decode(z_adv   * vae.config.scaling_factor).sample

def to_pil(t):
    t = (t.clamp(-1, 1) + 1) / 2
    t = (t[0].permute(1, 2, 0).cpu().float().numpy() * 255).astype(np.uint8)
    return Image.fromarray(t)

to_pil(recon_clean).save("recon_clean.png")
to_pil(recon_adv).save("recon_adv.png")
print("\nSaved: recon_clean.png / recon_adv.png")