import os
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def generate_masks(config):
    os.makedirs(config["masks_output_dir"], exist_ok=True)

    coco = COCO(config["annotations_path"])
    cat_ids = coco.getCatIds(catNms=[config["category"]])
    img_ids = coco.getImgIds(catIds=cat_ids)

    print(f"Immagini con '{config['category']}': {len(img_ids)}")

    skipped = 0
    saved   = 0

    for img_id in img_ids:
        img_info = coco.loadImgs(img_id)[0]
        h, w     = img_info["height"], img_info["width"]

        # Prendi tutte le annotazioni (iscrowd=0 e iscrowd=1)
        ann_ids = coco.getAnnIds(imgIds=img_id, catIds=cat_ids)
        anns    = coco.loadAnns(ann_ids)

        if not anns:
            if config["skip_no_person"]:
                skipped += 1
                continue

        # Costruisci maschera binaria unificata
        mask = np.zeros((h, w), dtype=np.uint8)
        for ann in anns:
            mask = np.maximum(mask, coco.annToMask(ann))

        # Converti in PIL con convenzione: nero=persona (0), bianco=sfondo (255)
        mask_pil = Image.fromarray((1 - mask) * 255).convert("L")

        # Salva con stesso nome dell'immagine ma estensione .png
        img_name    = os.path.splitext(img_info["file_name"])[0]
        output_path = os.path.join(config["masks_output_dir"], f"{img_name}.png")
        mask_pil.save(output_path)
        saved += 1

        if saved % 100 == 0:
            print(f"Salvate {saved}/{len(img_ids)} maschere...")

    print(f"\nDone. Maschere salvate: {saved} | Skippate (no person): {skipped}")

def test_single_image(config):
    os.makedirs(config["masks_output_dir"], exist_ok=True)

    coco    = COCO(config["annotations_path"])
    cat_ids = coco.getCatIds(catNms=[config["category"]])

    img_info = coco.loadImgs(config["test_img_id"])[0]
    h, w     = img_info["height"], img_info["width"]
    print(f"Immagine: {img_info['file_name']} ({w}x{h})")

    ann_ids = coco.getAnnIds(imgIds=config["test_img_id"], catIds=cat_ids)
    anns    = coco.loadAnns(ann_ids)
    print(f"Annotazioni trovate: {len(anns)}")

    # Costruisci maschera
    mask = np.zeros((h, w), dtype=np.uint8)
    for ann in anns:
        mask = np.maximum(mask, coco.annToMask(ann))

    # Converti: nero=persona (0), bianco=sfondo (255)
    mask_pil = Image.fromarray((1 - mask) * 255).convert("L")

    # Salva
    img_name    = os.path.splitext(img_info["file_name"])[0]
    output_path = os.path.join(config["masks_output_dir"], f"{img_name}.png")
    mask_pil.save(output_path)
    print(f"Maschera salvata in: {output_path}")

    # Visualizza immagine originale + maschera affiancate
    image_path = os.path.join(config["images_dir"], img_info["file_name"])
    image      = Image.open(image_path).convert("RGB")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(image);    axes[0].set_title("Immagine originale"); axes[0].axis("off")
    axes[1].imshow(mask_pil, cmap="gray"); axes[1].set_title("Maschera (nero=persona)"); axes[1].axis("off")
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(config["masks_output_dir"], f"{img_name}.png"))

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

config = {
    "images_dir":       "/andromeda/datasets/COCO/COCO2017_train/train2017",
    "annotations_path": "/andromeda/datasets/COCO/COCO2017_train/annotations/instances_train2017.json",  # <-- modifica
    "masks_output_dir": "/equilibrium/ldelbene/Immunization/data/COCO_mask/train",                         # <-- modifica
    "category":         "person",
    "test_img_id":      385029 ,  # <-- id immagine da testare (puoi cambiarlo)
}


if __name__ == "__main__":
   generate_masks(config)