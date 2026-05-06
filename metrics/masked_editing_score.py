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

    @staticmethod
    def _background_change_ssim(
            img1: Image.Image,
            img2: Image.Image,
            mask: np.ndarray,  # (H, W) binario float {0,1}: 1=background, 0=soggetto
    ) -> float:
        """
        Misura quanto il background (mask==1) è cambiato tra img1 e img2.

        Alto  → le due immagini hanno background simile (immunizzazione fallita)
        Basso → il background è stato disturbato (immunizzazione riuscita)
        """
        arr1 = MaskedEditingScore._to_numpy(img1).astype(np.float32) / 255.0
        arr2 = MaskedEditingScore._to_numpy(img2).astype(np.float32) / 255.0

        bg_mask = (mask == 1)  # True dove c'è background da editare

        if bg_mask.sum() == 0:
            return 1.0

        # estrai solo i pixel del background su entrambe le immagini
        # ricostruisce immagine con solo il background visibile, soggetto a zero
        mask_3ch = np.stack([bg_mask] * 3, axis=-1).astype(np.float32)
        arr1_bg = arr1 * mask_3ch
        arr2_bg = arr2 * mask_3ch

        raw_ssim = structural_similarity(
            arr1_bg, arr2_bg,
            channel_axis=2,
            data_range=1.0,
        )

        # de-bias: rimuove il contributo dei pixel azzerati (soggetto)
        bg_ratio = bg_mask.mean()
        corrected = (raw_ssim - (1.0 - bg_ratio)) / (bg_ratio + 1e-8)
        return float(np.clip(corrected, 0.0, 1.0))

    def _background_change_lpips(
            self,
            img1: Image.Image,
            img2: Image.Image,
            mask: np.ndarray,  # (H, W) binario float {0,1}: 1=background
    ) -> float:
        """
        LPIPS solo sul background (mask==1).

        Alto  → background molto diverso tra orig e adv (immunizzazione riuscita)
        Basso → background simile (immunizzazione fallita)
        """
        t1 = self._to_tensor(img1).to(self.device)
        t2 = self._to_tensor(img2).to(self.device)

        mask_t = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(self.device)  # (1,1,H,W)

        t1_bg = t1 * mask_t
        t2_bg = t2 * mask_t

        with torch.no_grad():
            score = self.lpips_fn(t1_bg, t2_bg).item()

        # normalizza per la proporzione di background
        bg_ratio = mask.mean()
        return score / (bg_ratio + 1e-8) if bg_ratio > 0 else 0.0

    # ── Metrica principale ───────────────────────────────────────────────────

    def compute(
            self,
            image_orig: Image.Image,
            image_edited: Image.Image,
            mask: Image.Image | np.ndarray,
            w_bg: float = 0.4,  # peso background disturbance
            w_ssim: float = 0.3,  # peso background ssim
            w_subject: float = 0.3,  # peso subject preservation
    ) -> dict:
        """
        Parameters
        ----------
        image_orig   : edited_orig_recovered  (editing sull'originale)
        image_edited : edited_adv_recovered   (editing sull'immunizzata)
        mask         : pixel neri=soggetto (deve restare uguale), pixel bianchi=background (deve cambiare)

        Returns
        -------
        dict con:
            - bg_lpips          : LPIPS sul background   (alto = background disturbato = protezione ok)
            - bg_ssim           : SSIM sul background    (basso = background disturbato = protezione ok)
            - subject_lpips     : LPIPS sul soggetto     (basso = soggetto preservato = immunizzazione pulita)
            - editing_score     : score combinato        (alto = protezione efficace E soggetto preservato)
            - mask_coverage     : % di pixel background  (zona bianca)
        """
        H, W = np.array(image_orig).shape[:2]
        mask_np = self._prepare_mask(mask, size=(H, W))

        # ── Background: deve essere disturbato ──────────────────────────────────
        bg_lpips = self._background_change_lpips(image_orig, image_edited, mask_np)
        bg_ssim = self._background_change_ssim(image_orig, image_edited, mask_np)

        # ── Soggetto: deve restare uguale ───────────────────────────────────────
        subject_mask = 1.0 - mask_np  # inverti: 1=soggetto, 0=background
        subject_lpips = self._background_change_lpips(  # riusa lo stesso metodo con maschera invertita
            image_orig, image_edited, subject_mask
        )

        # ── Editing score ────────────────────────────────────────────────────────
        # Normalizzazione lineare :
        #
        # bg_lpips      : [0, +∞) → normalizziamo su un range atteso [0, 2]
        # bg_ssim       : [0, 1]  → già normalizzato, lo invertiamo
        # subject_lpips : [0, +∞) → normalizziamo su un range atteso [0, 1]

        bg_lpips_norm = float(np.clip(bg_lpips / 1.0, 0.0, 1.0))  # atteso max ~1.0, clip a 1
        bg_ssim_norm = float(1.0 - bg_ssim)  # già in [0,1], invertito
        subject_preserved = float(np.clip(1.0 - subject_lpips / 0.5, 0.0, 1.0))  # atteso max ~0.5, invertito

        editing_score = (
                w_bg * bg_lpips_norm +  # background deve cambiare    → alto = buono
                w_ssim * bg_ssim_norm +  # background deve essere diverso → alto = buono
                w_subject * subject_preserved  # soggetto deve restare uguale → alto = buono
        )

        return {
            "bg_lpips": bg_lpips,
            "bg_ssim": bg_ssim,
            "subject_lpips": subject_lpips,
            "editing_score": editing_score,
            "mask_coverage": float(mask_np.mean()),
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