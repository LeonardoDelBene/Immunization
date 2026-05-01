import os
import warnings
import matplotlib.pyplot as plt
import torchvision.transforms as T
from PIL import ImageOps
from datasets import load_from_disk

from utils import (
    load_sample_from_hf,
    prepare_mask_and_masked_image,
    recover_image,
    set_seed_lib,
)
from model import Attack, AttackInstructPix2Pix, DiffVaxImmunization
from metrics import create_metric, MetricType

warnings.filterwarnings("ignore", message="QuickGELU mismatch", category=UserWarning, module="open_clip")




# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────

def get_output_dir(base_output_dir, use_instruct_pix2pix, run_wandb, sample_idx):
    subfolder = "InstructionPix2Pix" if use_instruct_pix2pix else "SD_Inpainting"
    if sample_idx == "full_dataset":
        output_dir = os.path.join(base_output_dir, subfolder, "full_dataset", run_wandb)
    else:
        output_dir = os.path.join(base_output_dir, subfolder, f"img_{sample_idx}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def load_models(config):
    print("Loading diffusion model and immunization checkpoint...")
    if config["use_instruct_pix2pix"]:
        attack_model = AttackInstructPix2Pix()
    else:
        attack_model = Attack(config["attack_model"])

    immunization_mdl = DiffVaxImmunization(
        load_existing=config["load_existing"],
        load_path=config["checkpoint_path"],
    )
    print("Done.")
    return attack_model, immunization_mdl


def load_sample(config):
    dataset = load_from_disk(config["dataset_path"])
    dataset_split = dataset[config["dataset_split"]]
    sample = dataset_split[config["sample_idx"]]
    image, image_mask = load_sample_from_hf(sample, split=config["dataset_split"])
    if config["run_full_dataset"]:
        edit_prompt = sample["prompts"][0]
        print(f"Prompt: {edit_prompt}")
    else:
        edit_prompt = config["edit_prompt"]
        print(f"Prompt: {config['edit_prompt']}")
    return image, image_mask, edit_prompt


# ─────────────────────────────────────────────
# IMMUNIZATION
# ─────────────────────────────────────────────

def immunize(image, image_mask, immunization_mdl, seed):
    to_pil = T.ToPILImage()
    mask_torch, image_torch, _ = prepare_mask_and_masked_image(image, image_mask)
    image_torch = image_torch.half().cuda()
    mask_torch  = mask_torch.half().cuda()

    set_seed_lib(seed)
    immunized_img = immunization_mdl.immunize_img(image_torch, mask_torch)

    adv_X = (immunized_img / 2 + 0.5).clamp(0, 1) # porta l'img in [0,1]
    adv_image_png = to_pil(adv_X[0]).convert("RGB")
    adv_image_png = recover_image(adv_image_png, image, image_mask, background=True)
    return adv_image_png


# ─────────────────────────────────────────────
# EDITING
# ─────────────────────────────────────────────

def edit_images(attack_model, image, adv_image_png, image_mask, edit_prompt, use_instruct_pix2pix):
    if use_instruct_pix2pix:
        edited_orig = attack_model.edit_image(edit_prompt, image)[0]
        edited_adv  = attack_model.edit_image(edit_prompt, adv_image_png)[0]
        edited_orig_recovered = edited_orig
        edited_adv_recovered  = edited_adv
    else:
        edited_orig = attack_model.edit_image(edit_prompt, image, image_mask)[0]
        edited_adv  = attack_model.edit_image(edit_prompt, adv_image_png, image_mask)[0]
        edited_orig_recovered = recover_image(edited_orig, image, image_mask, background=False)
        edited_adv_recovered  = recover_image(edited_adv,  adv_image_png, image_mask, background=False)

    print("Immunization and edits completed.")
    return edited_orig_recovered, edited_adv_recovered


# ─────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────

def save_images(output_dir, image, adv_image_png, edited_orig_recovered, edited_adv_recovered):
    image.save(os.path.join(output_dir, "original_image.png"))
    adv_image_png.save(os.path.join(output_dir, "immunized_image.png"))
    edited_orig_recovered.save(os.path.join(output_dir, "edited_original.png"))
    edited_adv_recovered.save(os.path.join(output_dir, "edited_immunized.png"))
    print(f"Images saved in {output_dir}")


def save_metrics(output_dir, edit_prompt, metrics: dict):
    prompt_file = os.path.join(output_dir, "prompt_and_metrics.txt")
    with open(prompt_file, "w") as f:
        f.write("--- Prompt ---\n")
        f.write(f"{edit_prompt}\n\n")

        f.write("--- Metrics ---\n\n")

        f.write("Image Quality (Original vs Immunized)\n")
        f.write(f"PSNR: {metrics['psnr_orig_adv']:.4f}\n")
        f.write(f"SSIM: {metrics['ssim_orig_adv']:.4f}\n")
        f.write(f"FSIM: {metrics['fsim_orig_adv']:.4f}\n\n")

        f.write("Protection Effectiveness (Edited Original vs Edited Immunized)\n")
        f.write(f"PSNR: {metrics['psnr_edit']:.4f}\n")
        f.write(f"SSIM: {metrics['ssim_edit']:.4f}\n")
        f.write(f"FSIM: {metrics['fsim_edit']:.4f}\n\n")

        f.write("CLIP score (Image vs Prompt)\n")
        f.write(f"Edited Original:  {metrics['clip_orig']:.4f}\n")
        f.write(f"Edited Immunized: {metrics['clip_adv']:.4f}\n\n")

        f.write("Caption Similarity (Edited Original vs Edited Immunized)\n")
        f.write(f"Score:        {metrics['caption_sim']:.4f}\n")
        f.write(f"Caption orig: {metrics['caption_orig']}\n")
        f.write(f"Caption adv:  {metrics['caption_adv']}\n\n")

        f.write("Accuracy Rate (Edited Original vs Edited Immunized)\n")
        f.write(f"LLM score: {metrics['accuracy_score']:.4f}\n")
        f.write(f"Success:   {metrics['accuracy_success']}\n")
        f.write(f"Accuracy:  {metrics['accuracy_rate']:.4f}\n\n")

        f.write("Masked Editing Score\n")
        f.write(f"Background LPIPS  (alto=protezione ok) : {metrics['masked_bg_lpips']:.4f}\n")
        f.write(f"Background SSIM   (basso=protezione ok): {metrics['masked_bg_ssim']:.4f}\n")
        f.write(f"Subject LPIPS     (basso=soggetto ok)  : {metrics['masked_subject_lpips']:.4f}\n")
        f.write(f"Editing score                          : {metrics['masked_editing_score']:.4f}\n")
        f.write(f"Mask coverage (% background)           : {metrics['masked_coverage']:.4f}\n")

    print(f"Metrics saved in {prompt_file}")

# ---------------------------------------------
# METRICS
# ---------------------------------------------

def load_metrics_models():
    print("Loading metric models...")
    metrics_models = {
        "psnr":    create_metric(MetricType.PSNR),
        "ssim":    create_metric(MetricType.SSIM),
        "fsim":    create_metric(MetricType.FSIM),
        "clip":    create_metric(MetricType.CLIP,   model="ViT-B-32", pretrained_on="openai"),
        "caption": create_metric(MetricType.CAP,    load_in_4bit=True),
        "accuracy": create_metric(MetricType.ACC,   load_in_4bit=True, threshold=0.5),
        "masked":  create_metric(MetricType.MASKED, lpips_net="alex"),
    }
    print("Metric models loaded.")
    return metrics_models


def compute_metrics(image, mask, adv_image_png, edited_orig_recovered, edited_adv_recovered, edit_prompt, metrics_models):
    psnr_metric    = metrics_models["psnr"]
    ssim_metric    = metrics_models["ssim"]
    fsim_metric    = metrics_models["fsim"]
    clip_metric    = metrics_models["clip"]
    caption_metric = metrics_models["caption"]
    accuracy_metric = metrics_models["accuracy"]
    masked_metric  = metrics_models["masked"]

    cap_score    = caption_metric.compute(edited_orig_recovered, edited_adv_recovered)
    acc_score    = accuracy_metric.compute(edited_orig_recovered, edited_adv_recovered)
    masked_score = masked_metric.compute(edited_orig_recovered, edited_adv_recovered, mask)

    metrics = {
        # Image quality
        "psnr_orig_adv": psnr_metric.calculate_metric_between_images(image, adv_image_png),
        "ssim_orig_adv": ssim_metric.calculate_metric_between_images(image, adv_image_png),
        "fsim_orig_adv": fsim_metric.calculate_metric_between_images(image, adv_image_png),
        # Protection effectiveness
        "psnr_edit":     psnr_metric.calculate_metric_between_images(edited_orig_recovered, edited_adv_recovered),
        "ssim_edit":     ssim_metric.calculate_metric_between_images(edited_orig_recovered, edited_adv_recovered),
        "fsim_edit":     fsim_metric.calculate_metric_between_images(edited_orig_recovered, edited_adv_recovered),
        # CLIP
        "clip_orig":     clip_metric.calculate_clip_score(edited_orig_recovered, edit_prompt),
        "clip_adv":      clip_metric.calculate_clip_score(edited_adv_recovered,  edit_prompt),
        # Caption similarity
        "caption_sim":   cap_score["caption_similarity"],
        "caption_orig":  cap_score["caption_1"],
        "caption_adv":   cap_score["caption_2"],
        # Accuracy rate
        "accuracy_rate":    acc_score["accuracy_rate"],
        "accuracy_score":   acc_score["llm_score"],
        "accuracy_success": acc_score["success"],
        # Masked editing score
        "masked_bg_lpips": masked_score["bg_lpips"],
        "masked_bg_ssim": masked_score["bg_ssim"],
        "masked_subject_lpips": masked_score["subject_lpips"],
        "masked_editing_score": masked_score["editing_score"],
        "masked_coverage": masked_score["mask_coverage"],
    }

    print("\n--- Image Quality (Original vs Immunized) ---")
    print(f"PSNR : {metrics['psnr_orig_adv']:.4f}")
    print(f"SSIM : {metrics['ssim_orig_adv']:.4f}")
    print(f"FSIM : {metrics['fsim_orig_adv']:.4f}")

    print("\n--- Protection Effectiveness (Edited Original vs Edited Immunized) ---")
    print(f"PSNR : {metrics['psnr_edit']:.4f}")
    print(f"SSIM : {metrics['ssim_edit']:.4f}")
    print(f"FSIM : {metrics['fsim_edit']:.4f}")

    print("\n--- CLIP score (Image vs Prompt) ---")
    print(f"Orig : {metrics['clip_orig']:.4f}")
    print(f"Adv  : {metrics['clip_adv']:.4f}")

    print("\n--- Caption Similarity (Edited Original vs Edited Immunized) ---")
    print(f"Score       : {metrics['caption_sim']:.4f}")
    print(f"Caption orig: {metrics['caption_orig']}")
    print(f"Caption adv : {metrics['caption_adv']}")

    print("\n--- Accuracy Rate (Edited Original vs Edited Immunized) ---")
    print(f"LLM score : {metrics['accuracy_score']:.4f}")
    print(f"Threshold : {accuracy_metric.threshold:.2f}")
    print(f"Success   : {metrics['accuracy_success']}")
    print(f"Accuracy  : {metrics['accuracy_rate']:.4f}")

    print("\n--- Masked Editing Score ---")
    print("\n--- Masked Editing Score ---")
    print(f"Background LPIPS  (alto=protezione ok) : {metrics['masked_bg_lpips']:.4f}")
    print(f"Background SSIM   (basso=protezione ok): {metrics['masked_bg_ssim']:.4f}")
    print(f"Subject LPIPS     (basso=soggetto ok)  : {metrics['masked_subject_lpips']:.4f}")
    print(f"Editing score                          : {metrics['masked_editing_score']:.4f}")
    print(f"Mask coverage (% background)           : {metrics['masked_coverage']:.4f}")

    return metrics

# ─────────────────────────────────────────────
# PLOT
# ─────────────────────────────────────────────

def plot_results(image, adv_image_png, edited_orig_recovered, edited_adv_recovered,
                 edit_prompt, use_instruct_pix2pix):
    model_label = "InstructPix2Pix" if use_instruct_pix2pix else "SD Inpainting"

    fig, axes = plt.subplots(1, 4, figsize=(15, 5))
    axes[0].imshow(image);                  axes[0].set_title("Original image");           axes[0].axis("off")
    axes[1].imshow(adv_image_png);          axes[1].set_title("Immunized image");          axes[1].axis("off")
    axes[2].imshow(edited_orig_recovered);  axes[2].set_title(f"Edited original [{model_label}]\n\"{edit_prompt}\""); axes[2].axis("off")
    axes[3].imshow(edited_adv_recovered);   axes[3].set_title(f"Edited immunized [{model_label}]\n\"{edit_prompt}\""); axes[3].axis("off")

    plt.suptitle(f"DiffVax: original vs immunized — {model_label}", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()




def run_on_full_dataset(config):
    """Esegue la pipeline completa su tutto il dataset di validation."""

    output_dir_base = get_output_dir(
        config["base_output_dir"],
        config["use_instruct_pix2pix"],
        config["run_wandb"],
        sample_idx="full_dataset"
    )

    attack_model, immunization_mdl = load_models(config)
    metrics_models = load_metrics_models()

    dataset = load_from_disk(config["dataset_path"])
    dataset = dataset[config["dataset_split"]]
    print(f"Dataset size: {len(dataset)} samples")

    all_metrics = []

    for sample_idx in range(len(dataset)):
        print(f"\n{'='*50}")
        print(f"Processing sample {sample_idx + 1}/{len(dataset)}")
        print(f"{'='*50}")

        try:
            # --- Carica sample ---
            sample = dataset[sample_idx]
            image, image_mask = load_sample_from_hf(sample, split=config["dataset_split"])
            edit_prompt = sample["prompts"][0]
            print(f"Prompt: {edit_prompt}")

            if not config["edit_background"]:
                image_mask = ImageOps.invert(image_mask)

            # --- Cartella output per questo sample ---
            sample_output_dir = os.path.join(output_dir_base, f"img_{sample_idx}")
            os.makedirs(sample_output_dir, exist_ok=True)

            # --- Immunizzazione ---
            adv_image_png = immunize(image, image_mask, immunization_mdl, config["seed"])

            # --- Editing ---
            edited_orig_recovered, edited_adv_recovered = edit_images(
                attack_model, image, adv_image_png, image_mask,
                edit_prompt, config["use_instruct_pix2pix"]
            )

            # --- Salvataggio immagini ---
            save_images(sample_output_dir, image, adv_image_png,
                        edited_orig_recovered, edited_adv_recovered)

            # --- Metriche ---
            metrics = compute_metrics(
                image,image_mask ,adv_image_png,
                edited_orig_recovered, edited_adv_recovered,
                edit_prompt,
                metrics_models
            )
            save_metrics(sample_output_dir, edit_prompt, metrics)

            # Accumula metriche per il summary finale
            all_metrics.append({"sample_idx": sample_idx, "prompt": edit_prompt, **metrics})

        except Exception as e:
            print(f"[ERROR] Sample {sample_idx} failed: {e}")
            continue

    # --- Summary globale ---
    save_global_summary(output_dir_base, all_metrics)




def save_global_summary(output_dir, all_metrics):
    """Salva un file di riepilogo con le metriche medie su tutto il dataset."""

    if not all_metrics:
        print("No metrics to summarize.")
        return

    numeric_keys = ["psnr_orig_adv", "ssim_orig_adv", "fsim_orig_adv",
                    "psnr_edit",     "ssim_edit",      "fsim_edit",
                    "clip_orig",     "clip_adv",        "caption_sim",
                    "accuracy_score",
                    "masked_bg_lpips", "masked_bg_ssim",
                    "masked_subject_lpips", "masked_editing_score", "masked_coverage"]

    averages = {k: sum(m[k] for m in all_metrics) / len(all_metrics) for k in numeric_keys}

    success_rate = sum(m["accuracy_success"] for m in all_metrics) / len(all_metrics) * 100

    summary_path = os.path.join(output_dir, "global_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Global Summary — {len(all_metrics)} samples\n")
        f.write("=" * 50 + "\n\n")

        f.write("Average Image Quality (Original vs Immunized)\n")
        f.write(f"PSNR: {averages['psnr_orig_adv']:.4f}\n")
        f.write(f"SSIM: {averages['ssim_orig_adv']:.4f}\n")
        f.write(f"FSIM: {averages['fsim_orig_adv']:.4f}\n\n")

        f.write("Average Protection Effectiveness (Edited Original vs Edited Immunized)\n")
        f.write(f"PSNR: {averages['psnr_edit']:.4f}\n")
        f.write(f"SSIM: {averages['ssim_edit']:.4f}\n")
        f.write(f"FSIM: {averages['fsim_edit']:.4f}\n\n")

        f.write("Average CLIP score\n")
        f.write(f"Edited Original:  {averages['clip_orig']:.4f}\n")
        f.write(f"Edited Immunized: {averages['clip_adv']:.4f}\n\n")

        f.write("Average Caption Similarity (Edited Original vs Edited Immunized)\n")
        f.write(f"Score: {averages['caption_sim']:.4f}\n\n")

        f.write("Accuracy Rate (Edited Original vs Edited Immunized)\n")
        f.write(f"Avg LLM score : {averages['accuracy_score']:.4f}\n")
        f.write(f"Success rate  : {success_rate:.1f}% ({sum(m['accuracy_success'] for m in all_metrics)}/{len(all_metrics)} samples above threshold)\n\n")

        f.write("Average Masked Editing Score\n")
        f.write(f"Background LPIPS  (alto=protezione ok) : {averages['masked_bg_lpips']:.4f}\n")
        f.write(f"Background SSIM   (basso=protezione ok): {averages['masked_bg_ssim']:.4f}\n")
        f.write(f"Subject LPIPS     (basso=soggetto ok)  : {averages['masked_subject_lpips']:.4f}\n")
        f.write(f"Editing score                          : {averages['masked_editing_score']:.4f}\n")
        f.write(f"Mask coverage (% background)           : {averages['masked_coverage']:.4f}\n\n")

        f.write("=" * 50 + "\n")
        f.write("Per-sample detail\n\n")
        for m in all_metrics:
            f.write(f"[{m['sample_idx']}] {m['prompt']}\n")
            f.write(f"  PSNR orig/adv: {m['psnr_orig_adv']:.4f} | edit: {m['psnr_edit']:.4f}\n")
            f.write(f"  SSIM orig/adv: {m['ssim_orig_adv']:.4f} | edit: {m['ssim_edit']:.4f}\n")
            f.write(f"  FSIM orig/adv: {m['fsim_orig_adv']:.4f} | edit: {m['fsim_edit']:.4f}\n")
            f.write(f"  CLIP orig: {m['clip_orig']:.4f} | adv: {m['clip_adv']:.4f}\n")
            f.write(f"  Caption similarity: {m['caption_sim']:.4f}\n")
            f.write(f"  Caption orig: {m['caption_orig']}\n")
            f.write(f"  Caption adv:  {m['caption_adv']}\n")
            f.write(f"  Accuracy LLM score: {m['accuracy_score']:.4f} | success: {m['accuracy_success']}\n")
            f.write(f"  Background LPIPS: {m['masked_bg_lpips']:.4f} | Background SSIM: {m['masked_bg_ssim']:.4f}\n")
            f.write(f"  Subject LPIPS: {m['masked_subject_lpips']:.4f} | Editing score: {m['masked_editing_score']:.4f}\n\n")

    print(f"\nGlobal summary saved in {summary_path}")

    print("\n" + "=" * 50)
    print(f"GLOBAL AVERAGES ({len(all_metrics)} samples)")
    print("=" * 50)
    print(f"PSNR orig/adv : {averages['psnr_orig_adv']:.4f}")
    print(f"SSIM orig/adv : {averages['ssim_orig_adv']:.4f}")
    print(f"FSIM orig/adv : {averages['fsim_orig_adv']:.4f}")
    print(f"PSNR edit     : {averages['psnr_edit']:.4f}")
    print(f"SSIM edit     : {averages['ssim_edit']:.4f}")
    print(f"FSIM edit     : {averages['fsim_edit']:.4f}")
    print(f"CLIP orig     : {averages['clip_orig']:.4f}")
    print(f"CLIP adv      : {averages['clip_adv']:.4f}")
    print(f"Caption sim   : {averages['caption_sim']:.4f}")
    print(f"Accuracy score: {averages['accuracy_score']:.4f}")
    print(f"Success rate  : {success_rate:.1f}% ({sum(m['accuracy_success'] for m in all_metrics)}/{len(all_metrics)} samples above threshold)")
    print(f"BG LPIPS      : {averages['masked_bg_lpips']:.4f}")
    print(f"BG SSIM       : {averages['masked_bg_ssim']:.4f}")
    print(f"Subject LPIPS : {averages['masked_subject_lpips']:.4f}")
    print(f"Editing score : {averages['masked_editing_score']:.4f}")
    print(f"Mask coverage : {averages['masked_coverage']:.4f}")


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

def get_config():
    return {
        "use_instruct_pix2pix": False,
        "edit_prompt":          "A person in a city parade",
        "seed":                 2043,
        "edit_background":      True,
        "load_existing":        True,
        "checkpoint_path":      os.path.join("checkpoints", "unet_best_2snl83u4.pth"),
        "attack_model":         "runwayml/stable-diffusion-inpainting",
        "base_output_dir":      "output",
        "dataset_path":         "./data/DiffVaxDataset_local",
        "dataset_split":        "validation",
        "sample_idx":           98,
        "run_full_dataset":     False,
        "run_wandb":            "DiffVax"
    }

def main():
    config = get_config()

    if config["run_full_dataset"]:
        run_on_full_dataset(config)
    else:
        output_dir = get_output_dir(
            config["base_output_dir"],
            config["use_instruct_pix2pix"],
            config["run_wandb"],
            config["sample_idx"]
        )
        attack_model, immunization_mdl = load_models(config)
        image, image_mask, _ = load_sample(config)
        if not config["edit_background"]:
            image_mask = ImageOps.invert(image_mask)
        adv_image_png = immunize(image, image_mask, immunization_mdl, config["seed"])
        edited_orig_recovered, edited_adv_recovered = edit_images(
            attack_model, image, adv_image_png, image_mask,
            config["edit_prompt"], config["use_instruct_pix2pix"]
        )
        save_images(output_dir, image, adv_image_png, edited_orig_recovered, edited_adv_recovered)
        metrics_model = load_metrics_models()
        metrics = compute_metrics(image, image_mask, adv_image_png, edited_orig_recovered, edited_adv_recovered,
                                  config["edit_prompt"], metrics_model)
        save_metrics(output_dir, config["edit_prompt"], metrics)
        plot_results(image, adv_image_png, edited_orig_recovered, edited_adv_recovered,
                     config["edit_prompt"], config["use_instruct_pix2pix"])


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    main()


