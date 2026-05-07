"""
Script per testare SD_inpainting con immagini locali immunizzate.
Supporta:
- Caricamento immagine locale già immunizzata
- Caricamento maschera locale (o creazione automatica)
- Editing con SD_inpainting
- Salvataggio risultati
"""

import os
import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw
from pathlib import Path

from utils import (
    prepare_mask_and_masked_image,
    recover_image,
    set_seed_lib,
    load_image_from_path,
)
from model import Attack, AttackInstructPix2Pix
from metrics import create_metric, MetricType


# ─────────────────────────────────────────────
# UTILITY FUNCTIONS
# ─────────────────────────────────────────────



def create_rectangular_mask(image_size=(512, 512), bbox=(100, 100, 400, 400)):
    """ Crea una maschera rettangolare. Args: image_size: (H, W) della maschera bbox: (x1, y1, x2, y2) coordinate del rettangolo Returns: PIL Image in scala di grigio (bianco = maschera) """
    mask = Image.new("L", (image_size[1], image_size[0]), 255)
    draw = ImageDraw.Draw(mask)
    draw.rectangle(bbox, fill=0)
    return mask


def load_image_and_mask(image_path, mask_path=None, image_size=(512, 512)):
    """
    Carica immagine e maschera da percorsi locali.
    
    Args:
        image_path: percorso immagine
        mask_path: percorso maschera (opzionale, se None crea maschera circolare)
        image_size: resize dell'immagine
    
    Returns:
        image (PIL), mask (PIL)
    """
    # Carica immagine
    image = load_image_from_path(image_path, size=image_size)
    print(f"✓ Immagine caricata: {image_path} ({image.size})")
    
    # Carica o crea maschera
    if mask_path and os.path.exists(mask_path):
        mask = Image.open(mask_path).convert("L").resize(image_size)
        print(f"✓ Maschera caricata: {mask_path}")
    else:
        print("⚠ Maschera non fornita, creazo maschera circolare di default...")
        mask = create_rectangular_mask(image_size=image_size, bbox=(100, 100, 400, 400))
        print(f"✓ Maschera circolare creata (raggio=80px, centro=(256,256))")
    
    return image, mask


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

def process_local_image(
    original_image_path,
    immunized_image_path,
    mask_path=None,
    edit_prompt="a person in a bakery",
    output_dir="output_local",
    seed=5,
    compute_metrics_flag=True,
    use_instruct_pix2pix=False,
):
    """
    Processa un'immagine originale e una immunizzata con SD_inpainting o InstructPix2Pix.
    
    Args:
        original_image_path: percorso immagine originale (non immunizzata)
        immunized_image_path: percorso immagine immunizzata
        mask_path: percorso maschera locale (opzionale, ignorato se use_instruct_pix2pix=True)
        edit_prompt: prompt per l'editing
        output_dir: cartella per i risultati
        seed: seed per riproducibilità
        compute_metrics_flag: se calcolare metriche (sempre True per confronto completo)
        use_instruct_pix2pix: se usare InstructPix2Pix invece di SD_inpainting
    """
    
    # Setup
    set_seed_lib(seed)
    os.makedirs(output_dir, exist_ok=True)
    
    model_name = "InstructPix2Pix" if use_instruct_pix2pix else "SD_inpainting"
    
    print("\n" + "="*60)
    print(f"PROCESSING ORIGINAL & IMMUNIZED IMAGES WITH {model_name.upper()}")
    print("="*60)
    
    # --- 1. Carica immagini e maschera ---
    print("\n[1/4] Loading images and mask...")
    
    # Carica immagine originale
    original_image = load_image_from_path(original_image_path, size=(512,512))
    print(f"✓ Original image loaded: {original_image_path} ({original_image.size})")
    
    # Carica immagine immunizzata
    immunized_image = load_image_from_path(immunized_image_path, size=(512,512))
    print(f"✓ Immunized image loaded: {immunized_image_path} ({immunized_image.size})")
    
    # Carica maschera solo se non usi InstructPix2Pix
    if use_instruct_pix2pix:
        print("⚠ Using InstructPix2Pix (mask will be ignored)")
        mask = None
    else:
        if mask_path and os.path.exists(mask_path):
            mask = Image.open(mask_path).convert("L").resize((224,224))
            print(f"✓ Maschera caricata: {mask_path}")
        else:
            print("⚠ Maschera non fornita, creo maschera circolare di default...")
            mask = create_rectangular_mask(image_size=(512,512), bbox=(100, 100, 400, 400))
            print(f"✓ Maschera circolare creata (raggio=80px, centro=(256,256))")
    
    # --- 2. Carica modello ---
    print(f"\n[2/4] Loading {model_name} model...")
    try:
        if use_instruct_pix2pix:
            attack_model = AttackInstructPix2Pix()
            print(f"✓ {model_name} model loaded")
        else:
            attack_model = Attack("runwayml/stable-diffusion-inpainting")
            print(f"✓ {model_name} model loaded")
    except Exception as e:
        print(f"✗ Error loading {model_name}: {e}")
        return
    
    # --- 3. Editing ---
    print(f"\n[3/4] Editing with {model_name}...")
    try:
        if use_instruct_pix2pix:
            edited_original = attack_model.edit_image(edit_prompt, original_image)[0]
            edited_immunized = attack_model.edit_image(edit_prompt, immunized_image)[0]
            edited_original_recovered = edited_original
            edited_immunized_recovered = edited_immunized
        else:
            edited_original = attack_model.edit_image(edit_prompt, original_image, mask)[0]
            edited_immunized = attack_model.edit_image(edit_prompt, immunized_image, mask)[0]
            edited_original_recovered = recover_image(edited_original, original_image, mask, background=False)
            edited_immunized_recovered = recover_image(edited_immunized, immunized_image, mask, background=False)
        
        print("✓ Editing completed")
    except Exception as e:
        print(f"✗ Error during editing: {e}")
        return
    
    # --- 4. Salvataggio risultati ---
    print("\n[4/4] Saving results...")
    try:
        original_image.save(os.path.join(output_dir, "01_original_image.png"))
        immunized_image.save(os.path.join(output_dir, "02_immunized_image.png"))
        edited_original_recovered.save(os.path.join(output_dir, "03_edited_original.png"))
        edited_immunized_recovered.save(os.path.join(output_dir, "04_edited_immunized.png"))
        if mask is not None:
            mask.save(os.path.join(output_dir, "05_mask.png"))
        
        # Salva metadati
        with open(os.path.join(output_dir, "metadata.txt"), "w") as f:
            f.write("Original & Immunized Image Processing Results\n")
            f.write("="*50 + "\n\n")
            f.write(f"Original image path: {original_image_path}\n")
            f.write(f"Immunized image path: {immunized_image_path}\n")
            f.write(f"Mask path: {mask_path if mask_path else 'Auto-generated (circular)' if not use_instruct_pix2pix else 'N/A (not used by InstructPix2Pix)'}\n")
            f.write(f"Edit prompt: {edit_prompt}\n")
            f.write(f"Model: {model_name}\n")
            f.write(f"Seed: {seed}\n")
        
        print(f"✓ Results saved in: {output_dir}")
        print(f"  - 01_original_image.png")
        print(f"  - 02_immunized_image.png")
        print(f"  - 03_edited_original.png")
        print(f"  - 04_edited_immunized.png")
        if mask is not None:
            print(f"  - 05_mask.png")
        print(f"  - metadata.txt")
        
    except Exception as e:
        print(f"✗ Error saving results: {e}")
        return
    
    # --- 5. Metriche (sempre calcolate per confronto completo) ---
    print("\n[5/5] Computing all metrics...")
    try:
        psnr_metric = create_metric(MetricType.PSNR)
        ssim_metric = create_metric(MetricType.SSIM)
        fsim_metric = create_metric(MetricType.FSIM)
        clip_metric = create_metric(MetricType.CLIP, model="ViT-B-32", pretrained_on="openai")
        
        # Prova a creare il metrico CAP, se fallisce continua senza
        try:
            caption_metric = create_metric(MetricType.CAP, load_in_4bit=True)
            use_caption_metric = True
        except Exception as cap_error:
            print(f"⚠ Warning: Could not load CAP metric: {cap_error}")
            print("Continuing without caption similarity calculation...")
            use_caption_metric = False

        if use_caption_metric:
            cap_score = caption_metric.compute(edited_original_recovered, edited_immunized_recovered)
            caption_sim = cap_score["caption_similarity"]
            caption_orig = cap_score["caption_1"]
            caption_adv = cap_score["caption_2"]
        else:
            caption_sim = 0.0
            caption_orig = "N/A (metric not available)"
            caption_adv = "N/A (metric not available)"
        
        metrics = {
            "psnr_orig_adv": psnr_metric.calculate_metric_between_images(original_image, immunized_image),
            "ssim_orig_adv": ssim_metric.calculate_metric_between_images(original_image, immunized_image),
            "fsim_orig_adv": fsim_metric.calculate_metric_between_images(original_image, immunized_image),
            "psnr_edit":     psnr_metric.calculate_metric_between_images(edited_original_recovered, edited_immunized_recovered),
            "ssim_edit":     ssim_metric.calculate_metric_between_images(edited_original_recovered, edited_immunized_recovered),
            "fsim_edit":     fsim_metric.calculate_metric_between_images(edited_original_recovered, edited_immunized_recovered),
            "clip_orig":     clip_metric.calculate_clip_score(edited_original_recovered, edit_prompt),
            "clip_adv":      clip_metric.calculate_clip_score(edited_immunized_recovered,  edit_prompt),
            "caption_sim":   caption_sim,
            "caption_orig": caption_orig,
            "caption_adv": caption_adv,
        }
        
        # Salva metriche
        with open(os.path.join(output_dir, "metrics.txt"), "w") as f:
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
            
            f.write("Caption similarity (Edited Original vs Edited Immunized)\n")
            f.write(f"Score: {metrics['caption_sim']:.4f}\n")
            f.write("Caption orig : " + str(metrics['caption_orig']) + "\n")
            f.write("Caption adv : " + str(metrics['caption_adv']))
        
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
        
        print("\n--- Caption similarity (Edited Original vs Edited Immunized) ---")
        print(f"score : {metrics['caption_sim']:.4f}")
        print("caption orig : " + str(metrics['caption_orig']))
        print("caption adv : " + str(metrics['caption_adv']))
        
        print(f"\n✓ All metrics saved in: {os.path.join(output_dir, 'metrics.txt')}")
        
    except Exception as e:
        print(f"⚠ Error computing metrics: {e}")
        print("Continuing without metrics...")
    
    print("\n" + "="*60)
    print("✓ PROCESSING COMPLETE")
    print("="*60 + "\n")


# ─────────────────────────────────────────────
# EXAMPLES
# ─────────────────────────────────────────────

if __name__ == "__main__":
    
    # ============================================
    # EXAMPLE 1: SD_inpainting con confronto completo
    # ============================================
    process_local_image(
        original_image_path="./120.png",  # ← CAMBIA CON IL TUO PERCORSO
        immunized_image_path="./120_noise.png",  # ← CAMBIA CON IL TUO PERCORSO
        output_dir="output/SD_Inpainting/FOA/120",
        edit_prompt="a person in a garden",
        use_instruct_pix2pix=False,
    )


