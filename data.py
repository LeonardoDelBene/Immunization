import torch
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10
from datasets import load_from_disk
from PIL import Image, ImageOps
from pathlib import Path
from utils import load_sample_from_hf, prepare_mask_and_masked_image



class OxfordPetLocal(Dataset):
    """
    Carica Oxford-Pet da cartella locale con struttura:
    Oxford-Pet/
        train/
            img/
            mask/
        validation/
            img/
            mask/
    """

    def __init__(self, root: str, split: str = "train", image_size: int = 224):
        self.image_size = image_size

        split_folder = "validation" if split == "val" else split
        img_dir = Path(root) / split_folder / "img"
        mask_dir = Path(root) / split_folder / "mask"

        self.img_paths = sorted(img_dir.glob("*"))
        self.mask_paths = sorted(mask_dir.glob("*"))

        assert len(self.img_paths) == len(self.mask_paths), \
            f"Mismatch: {len(self.img_paths)} immagini vs {len(self.mask_paths)} maschere"

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx: int):
        image = Image.open(self.img_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L")

        return image, mask

class ImmunizationDataset(Dataset):
    def __init__(
        self,
        dataset:        str = "DiffVax",  # DiffVax | Oxford-Pet
        split:          str = "train",
        target:          str = "./data/target.png",
        image_size:     int = 224,
    ):
        self.split      = split
        self.image_size = image_size
        self.target = target

        if dataset == "DiffVax":
            dataset = load_from_disk("data/DiffVaxDataset_local")
            self.dataset = dataset[split]
        elif dataset == "Oxford-Pet":
            self.dataset = OxfordPetLocal(root="./data/Oxford-Pet", split=split, image_size=image_size)
        else:
            raise ValueError(f"dataset non supportato: {dataset}")




        # ── Transform per l'immagine target ──
        # prepare_mask_and_masked_image gestisce già I e M → [-1,1] e [0,1]
        # serve solo per il target
        self.target_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                 [0.5, 0.5, 0.5]),  # → [-1, 1]
        ])

        self.image_transform = transforms.Resize(
            (image_size, image_size),
            interpolation=InterpolationMode.BILINEAR
        )

        self.mask_transform = transforms.Resize(
            (image_size, image_size),
            interpolation=InterpolationMode.NEAREST
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        if isinstance(self.dataset, OxfordPetLocal):
            image, mask = self.dataset[idx]
        else:
            sample = self.dataset[idx]
            image, mask = load_sample_from_hf(sample, split=self.split)

        image = self.image_transform(image)
        mask = self.mask_transform(mask)

        mask = ImageOps.invert(mask)

        M, _, I = prepare_mask_and_masked_image(image, mask)
        I = I.squeeze(0) # rimuove dimensione batch aggiunta da prepare_mask
        M = M.squeeze(0)
        I_target = self.target_transform(Image.open(self.target).convert("RGB"))  # target è l'immagine originale trasformata

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

    titles = ["I  (immunizzata)", "M  (maschera)", "I_target  (coco)"]
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

    train_dataset = ImmunizationDataset(dataset="Oxford-Pet",split="train")
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)

    I, M, I_target = next(iter(train_loader))
    print(f"I       : {I.shape}        range [{I.min():.2f}, {I.max():.2f}]")
    print(f"M       : {M.shape}        range [{M.min():.2f}, {M.max():.2f}]")
    print(f"I_target: {I_target.shape} range [{I_target.min():.2f}, {I_target.max():.2f}]")

    show_batch_example(I, M, I_target, idx=0)  # ← cambia idx per altri esempi
