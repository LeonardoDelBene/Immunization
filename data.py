import torch
import numpy as np
import random
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10
from datasets import load_from_disk
from PIL import Image
from pathlib import Path

from utils import load_sample_from_hf, prepare_mask_and_masked_image, set_seed_lib


class SimpleImageFolder(Dataset):
    """Carica immagini da una cartella senza annotazioni."""
    
    def __init__(self, root_dir: str, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.image_paths = sorted(self.root_dir.glob("*.jpg")) + sorted(self.root_dir.glob("*.png"))
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img


class ImmunizationDataset(Dataset):
    """
    Dataset per il training di DiffVax.
    - Immagine + maschera: da ozdentarikcan/DiffVaxDataset  (via load_sample_from_hf)
    - Immagine target:     da CIFAR-10 o COCO (random sample)
    """

    def __init__(
        self,
        split:          str = "train",
        target_dataset: str = "cifar10",  # "cifar10" | "coco"
        coco_root:      str = "/andromeda/datasets/COCO/COCO2017_val/val2017",
        image_size:     int = 512,
        seed:           int = 5,
    ):
        self.split      = split
        self.image_size = image_size

        # ── Carica DiffVaxDataset ──
        dataset = load_from_disk("DiffVaxDataset_local")
        self.dataset = dataset[split]

        # ── Transform per l'immagine target ──
        # prepare_mask_and_masked_image gestisce già I e M → [-1,1] e [0,1]
        # serve solo per il target
        self.target_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                 [0.5, 0.5, 0.5]),  # → [-1, 1]
        ])

        # ── Carica dataset target ──
        self.target_dataset_name = target_dataset

        if target_dataset == "cifar10":
            self.target_data = CIFAR10(
                root="data/cifar10",
                train=(split == "train"),
                download=True,
                transform=self.target_transform,
            )
        elif target_dataset == "coco":
            if coco_root is None:
                raise ValueError("coco_root è necessario per COCO")
            self.target_data = SimpleImageFolder(
                root_dir=coco_root,
                transform=self.target_transform,
            )
        else:
            raise ValueError(f"target_dataset non supportato: {target_dataset}")


    def _get_target_image(self, idx: int) -> torch.Tensor:
        """
        Ritorna un'immagine target con mapping 1-to-1 in base all'indice.
        Immagine 0 → target 0, immagine 1 → target 1, ecc.
        """
        target_idx = idx % len(self.target_data)  # wrap around se necessario

        if self.target_dataset_name == "cifar10":
            img, _ = self.target_data[target_idx]  # CIFAR10 ritorna (img, label)
        elif self.target_dataset_name == "coco":
            img = self.target_data[target_idx]  # SimpleImageFolder ritorna solo img
            if isinstance(img, torch.Tensor) and img.shape[0] == 1:
                img = img.repeat(3, 1, 1)  # grayscale → RGB

        return img  # (3, H, W) in [-1, 1]

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        sample = self.dataset[idx]

        # ── Immagine e maschera tramite load_sample_from_hf ──
        image, mask = load_sample_from_hf(sample, split=self.split)

        # ── Prepara tensori con prepare_mask_and_masked_image ──
        # M     : (1, H, W)  in [0, 1]  binaria
        # I     : (1, 3, H, W) in [-1, 1]
        M, _, I = prepare_mask_and_masked_image(image, mask)

        # Rimuove la dimensione batch aggiunta da prepare_mask_and_masked_image
        I = I.squeeze(0)  # (3, H, W)
        M = M.squeeze(0)  # (1, H, W)

        # ── Target con mapping 1-to-1 per indice ──
        I_target = self._get_target_image(idx)  # (3, H, W)

        return I, M, I_target


import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def show_batch_example(I, M, I_target, idx=0):
    """
    Mostra un esempio del batch: immagine originale, maschera e immagine target.

    Args:
        I        : tensore (B, C, H, W) – immagine immunizzata
        M        : tensore (B, 1, H, W) o (B, H, W) – maschera binaria
        I_target : tensore (B, C, H, W) – immagine target (es. CIFAR-10)
        idx      : indice dell'esempio nel batch (default 0)
    """

    def to_numpy(t):
        """Converti tensore in numpy HxWxC, clip in [0,1]."""
        t = t.detach().cpu().float()
        if t.ndim == 3:  # C x H x W
            t = t.permute(1, 2, 0)
        if t.shape[-1] == 1:  # maschera grigio
            t = t.squeeze(-1)
        return t.clamp(0, 1).numpy()

    img = to_numpy(I[idx])
    mask = to_numpy(M[idx])
    target = to_numpy(I_target[idx])

    fig = plt.figure(figsize=(10, 3.5))
    gs = gridspec.GridSpec(1, 3, wspace=0.05)

    titles = ["I  (immunizzata)", "M  (maschera)", "I_target  (CIFAR-10)"]
    images = [img, mask, target]
    cmaps = [None, "gray", None]

    for col, (ax_title, data, cmap) in enumerate(zip(titles, images, cmaps)):
        ax = fig.add_subplot(gs[col])
        ax.imshow(data, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(ax_title, fontsize=9, pad=6)
        ax.axis("off")

    fig.suptitle(
        f"Batch sample #{idx}  —  "
        f"I: {tuple(I[idx].shape)}   "
        f"M: {tuple(M[idx].shape)}   "
        f"I_target: {tuple(I_target[idx].shape)}",
        fontsize=8, y=1.02
    )
    plt.tight_layout()
    plt.savefig("batch_sample.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Figura salvata in batch_sample.png")


# ── utilizzo nel main ───────────────────────────────────────────────────────
if __name__ == "__main__":
    from torch.utils.data import DataLoader

    train_dataset = ImmunizationDataset(split="train", target_dataset="coco")
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)

    I, M, I_target = next(iter(train_loader))
    print(f"I       : {I.shape}        range [{I.min():.2f}, {I.max():.2f}]")
    print(f"M       : {M.shape}        range [{M.min():.2f}, {M.max():.2f}]")
    print(f"I_target: {I_target.shape} range [{I_target.min():.2f}, {I_target.max():.2f}]")

    show_batch_example(I, M, I_target, idx=0)  # ← cambia idx per altri esempi