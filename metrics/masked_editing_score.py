# masked_editing_score.py

import torch
import torch.nn.functional as F
import numpy as np
import lpips
from PIL import Image
from skimage.metrics import structural_similarity
from .base import Metric


class MaskedEditingScore(Metric):
    """
    Masked Editing Score.

    Valuta la qualità dell'editing separando la regione editata dal background:

        edit_change            : LPIPS nella regione della maschera
                                 (alto = l'editing ha modificato la zona target)
        background_preservation: SSIM fuori dalla maschera
                                 (alto = il background è stato preservato)
        editing_score          : media pesata dei due
                                 (alto = editing avvenuto E background preservato)

    Nel contesto dell'immunizzazione:
        - immagine originale editata     → edit_change alto,   bg_preservation alto
        - immagine immunizzata editata   → edit_change basso,  bg_preservation alto
    """

    def __init__(self, *args, lpips_net: str = "alex", **kwargs):
        super().__init__(*args, **kwargs)
        self.device   = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.lpips_fn = lpips.LPIPS(net=lpips_net).to(self.device)
        self.lpips_fn.eval()

    # ── Conversioni ──────────────────────────────────────────────────────────

    @staticmethod
    def _to_numpy(image: Image.Image) -> np.ndarray:
        """PIL → numpy uint8 (H, W, 3)."""
        return np.array(image.convert("RGB"))

    @staticmethod
    def _to_tensor(image: Image.Image) -> torch.Tensor:
        """PIL → tensor [-1, 1] (1, 3, H, W)."""
        arr = np.array(image.convert("RGB")).astype(np.float32) / 127.5 - 1.0
        return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)

    @staticmethod
    def _prepare_mask(mask: Image.Image | np.ndarray, size: tuple) -> np.ndarray:
        if isinstance(mask, Image.Image):
            # gestisce sia L che RGB (nel tuo pipeline arriva RGB)
            mask = np.array(mask.convert("L"))
        mask = mask.astype(np.float32)
        if mask.max() > 1.0:
            mask = mask / 255.0
        return (mask > 0.5).astype(np.float32)

    # ── LPIPS mascherato ─────────────────────────────────────────────────────

    def _masked_lpips(
        self,
        img1: Image.Image,
        img2: Image.Image,
        mask: np.ndarray,           # (H, W) binario float
    ) -> float:
        """LPIPS calcolato solo nella regione mask==1."""
        t1 = self._to_tensor(img1).to(self.device)
        t2 = self._to_tensor(img2).to(self.device)

        H, W = mask.shape
        mask_t = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(self.device)  # (1,1,H,W)

        # applica la maschera a entrambe le immagini
        t1_masked = t1 * mask_t
        t2_masked = t2 * mask_t

        with torch.no_grad():
            score = self.lpips_fn(t1_masked, t2_masked).item()

        # normalizza per la proporzione di pixel mascherati
        mask_ratio = mask.mean()
        return score / (mask_ratio + 1e-8) if mask_ratio > 0 else 0.0

    # ── SSIM mascherato ──────────────────────────────────────────────────────

    @staticmethod
    def _masked_ssim(
            img1: Image.Image,
            img2: Image.Image,
            mask: np.ndarray,  # (H, W) binario float {0,1}
    ) -> float:
        """SSIM calcolato solo nella regione mask==0 (background)."""
        arr1 = MaskedEditingScore._to_numpy(img1).astype(np.float32) / 255.0
        arr2 = MaskedEditingScore._to_numpy(img2).astype(np.float32) / 255.0

        bg_mask = (mask == 0)  # True dove c'è background

        if bg_mask.sum() == 0:
            return 1.0

        # azzera la regione della maschera su entrambe → SSIM solo sul background
        mask_3ch = np.stack([bg_mask] * 3, axis=-1).astype(np.float32)
        arr1_bg = arr1 * mask_3ch
        arr2_bg = arr2 * mask_3ch

        return structural_similarity(
            arr1_bg, arr2_bg,
            channel_axis=2,
            data_range=1.0,
        )

    # ── Metrica principale ───────────────────────────────────────────────────

    def compute(
        self,
        image_orig:   Image.Image,
        image_edited: Image.Image,
        mask:         Image.Image | np.ndarray,
        w_edit:       float = 0.5,    # peso di edit_change nell'editing_score
        w_bg:         float = 0.5,    # peso di background_preservation
    ) -> dict:
        """
        Parameters
        ----------
        image_orig   : immagine prima dell'editing.
        image_edited : immagine dopo l'editing.
        mask         : maschera della regione editata (bianco=zona editata).
        w_edit       : peso dell'edit_change nell'editing_score finale.
        w_bg         : peso della background_preservation nell'editing_score finale.

        Returns
        -------
        dict con:
            - edit_change             : LPIPS nella zona editata  ∈ [0, +∞)  (alto = editing avvenuto)
            - background_preservation : SSIM nel background       ∈ [0,  1]  (alto = bg preservato)
            - editing_score           : score combinato           ∈ [0,  1]
            - mask_coverage           : % di pixel mascherati
        """
        H, W = np.array(image_orig).shape[:2]
        mask_np = self._prepare_mask(mask, (H, W))

        edit_change  = self._masked_lpips(image_orig, image_edited, mask_np)
        bg_preserv   = self._masked_ssim(image_orig, image_edited, mask_np)
        mask_coverage = mask_np.mean()

        # editing_score: normalizza edit_change in [0,1] con sigmoide,
        # poi media pesata con bg_preservation
        edit_change_norm = float(1 / (1 + np.exp(-edit_change + 1)))  # sigmoide centrata su 1
        editing_score    = w_edit * edit_change_norm + w_bg * bg_preserv

        return {
            "edit_change":             edit_change,
            "background_preservation": bg_preserv,
            "editing_score":           editing_score,
            "mask_coverage":           float(mask_coverage),
        }

    def calculate_metric_between_images(
        self,
        image_orig:   Image.Image,
        image_edited: Image.Image,
        mask:         Image.Image | np.ndarray,
    ) -> float:
        """Interfaccia semplificata — restituisce solo l'editing_score."""
        return self.compute(image_orig, image_edited, mask)["editing_score"]

    def __call__(
        self,
        image_orig:   Image.Image,
        image_edited: Image.Image,
        mask:         Image.Image | np.ndarray,
    ) -> dict:
        return self.compute(image_orig, image_edited, mask)


from utils import *
from datasets import load_from_disk

if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent))

    from metrics.masked_editing_score import MaskedEditingScore
    from PIL import Image
    import numpy as np
    # Esempio di utilizzo
    metric = MaskedEditingScore()
    img1 = Image.open("output/SD_Inpainting/img_0/edited_original.png")
    img2 = Image.open("output/SD_Inpainting/img_0/edited_immunized.png")

    dataset = load_from_disk("./data/DiffVaxDataset_local")
    dataset_split = dataset["validation"]
    sample = dataset_split[0]
    image, mask = load_sample_from_hf(sample, split="validation")

    results = metric(img1, img2, mask)
    print(results)