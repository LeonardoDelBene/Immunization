import torch
import numpy as np
import random
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10, CocoDetection
from datasets import load_from_disk
from PIL import Image

from utils import load_sample_from_hf, prepare_mask_and_masked_image, set_seed_lib


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
        coco_root:      str = None,
        coco_annot:     str = None,
        image_size:     int = 512,
        seed:           int = 5,
    ):
        self.split      = split
        self.image_size = image_size

        # ── Carica DiffVaxDataset ──
        dataset = load_from_disk("./DiffVaxDataset_local")
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
            if coco_root is None or coco_annot is None:
                raise ValueError("coco_root e coco_annot sono necessari per COCO")
            self.target_data = CocoDetection(
                root=coco_root,
                annFile=coco_annot,
                transform=self.target_transform,
            )
        else:
            raise ValueError(f"target_dataset non supportato: {target_dataset}")
        self.rng = np.random.default_rng(seed)


    def _get_target_image(self) -> torch.Tensor:
        """Ritorna un'immagine target random dal dataset target."""
        idx = int(self.rng.integers(0, len(self.target_data)))

        if self.target_dataset_name == "cifar10":
            img, _ = self.target_data[idx]
        elif self.target_dataset_name == "coco":
            img, _ = self.target_data[idx]
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

        # ── Target ──
        I_target = self._get_target_image()  # (3, H, W)

        return I, M, I_target

if __name__ == "__main__":
    from torch.utils.data import DataLoader

    train_dataset = ImmunizationDataset(split="train", target_dataset="cifar10")
    val_dataset   = ImmunizationDataset(split="validation", target_dataset="cifar10")

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_dataset,   batch_size=4, shuffle=False, num_workers=4)

    # Verifica un batch
    I, M, I_target = next(iter(train_loader))
    print(f"I       : {I.shape}        range [{I.min():.2f}, {I.max():.2f}]")
    print(f"M       : {M.shape}        range [{M.min():.2f}, {M.max():.2f}]")
    print(f"I_target: {I_target.shape} range [{I_target.min():.2f}, {I_target.max():.2f}]")