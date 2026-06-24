import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from typing import List, Tuple
from sklearn.cluster import KMeans
import lpips
import random



# ─────────────────────────────────────────────
# 1. LOSS DI IMPERCETTIBILITÀ
# ─────────────────────────────────────────────

def noise_loss(I_im, I, M, noise_on_mask=False):
    diff = I_im - I
    if noise_on_mask:
        diff = diff * M
    return diff.abs().mean()


# --------------------------------------------
# 2. LOSS LPAA PATCH
#---------------------------------------------

def lpaa_patch_loss(
    vae,
    img_orig: torch.Tensor,
    delta_total: torch.Tensor,
    posterior_target,
    n_masks: int = 5,
    patch_size: int = 16,
    keep_frac: float = 0.5,
    clamp_min: float = -1.0,
    clamp_max: float = 1.0,
):
    """
    LPAA: Local Patch Anti-Overfitting Alignment loss.

    Obiettivo:
    Forzare la perturbazione delta_total a essere efficace anche quando
    applicata solo a patch casuali dell'immagine.
    """

    B, C, H, W = img_orig.shape
    device = img_orig.device

    total_loss = 0.0

    for _ in range(n_masks):
        # ---- 1. mask patch random ----
        mask = torch.zeros((B, 1, H, W), device=device)

        ph = patch_size
        pw = patch_size

        # scegli top-left random per patch
        if H <= ph or W <= pw:
            # fallback: nessuna mask
            mask[:] = 1.0
        else:
            y = random.randint(0, H - ph)
            x = random.randint(0, W - pw)

            mask[:, :, y:y+ph, x:x+pw] = 1.0

        # opzionale: keep fraction (drop parte della patch)
        if keep_frac < 1.0:
            rand_keep = (torch.rand_like(mask) < keep_frac).float()
            mask = mask * rand_keep

        # ---- 2. applica perturbazione solo sulla patch ----
        img_masked = img_orig + delta_total * mask
        img_masked = torch.clamp(img_masked, clamp_min, clamp_max)

        # ---- 3. encode VAE ----
        posterior = vae.encode(img_masked).latent_dist

        # ---- 4. loss nello spazio latente ----
        loss = vae_mse(posterior, posterior_target)

        total_loss += loss

    return total_loss / n_masks

# ─────────────────────────────────────────────
# 3. LOSS VAE
# ─────────────────────────────────────────────
def vae_align_loss(posterior_im, posterior_target) -> Tensor:
    mu_im,  lv_im  = posterior_im.mean,    posterior_im.logvar
    mu_tgt, lv_tgt = posterior_target.mean, posterior_target.logvar

    # Clamp logvar per evitare exp() che esplode o var ≈ 0
    lv_im  = lv_im.clamp(-10, 10)
    lv_tgt = lv_tgt.clamp(-10, 10)

    #l_mu = F.mse_loss(mu_im, mu_tgt)

    var_im  = lv_im.exp()
    var_tgt = lv_tgt.exp().clamp(min=1e-6)  # evita divisione per zero

    kl = 0.5 * (
          var_im / var_tgt
        + (mu_tgt - mu_im).pow(2) / var_tgt
        + lv_tgt - lv_im
        - 1
    )

    # Clamp la KL per evitare valori esplosivi nel backward
    # log1p: gradiente sempre vivo, cresce lentamente per valori grandi
    l_kl = torch.log1p(kl.clamp(min=0)).mean()

    return l_kl


def vae_mse(posterior_im, posterior_target) -> Tensor:
    loss = F.mse_loss(
        posterior_im.mean,
        posterior_target.mean.expand_as(posterior_im.mean)
    )
    return loss

def vae_divergence_loss(
    p: "DiagonalGaussianDistribution",
    q: "DiagonalGaussianDistribution",
) -> torch.Tensor:
    """
    Distanza simmetrica tra due posterior gaussiani.
    Combina MSE sulle medie + MSE sui log-var per una metrica robusta.
    Restituisce un valore POSITIVO → più è alto, più i posterior sono distanti.
    """
    mse_mean   = F.mse_loss(p.mean,    q.mean)
    mse_logvar = F.mse_loss(p.logvar,  q.logvar)
    return mse_mean + mse_logvar



# ─────────────────────────────────────────────
# 4. PESATURA DINAMICA
# ─────────────────────────────────────────────

class DynamicWeighter:
    """
    Calcola i pesi adattativi W_i per l'ensemble di surrogati.

    S_i = L_i(t) / mean(L_i(t-N:t))   ← confronto con media mobile su finestra
    W_i = W_init * (n * exp(S_i / T_temp)) / sum_j(exp(S_j / T_temp))

    Rispetto alla versione originale:
    - S_i calcolato su finestra lunga (window) invece del solo passo precedente
    - T_temp bassa per amplificare le differenze piccole
    - S_i clippato per stabilità
    """

    def __init__(
        self,
        n_surrogates: int,
        W_init:  float = 1.0,
        T_temp:  float = 0.1,    # abbassato da 1.0 per amplificare differenze piccole
        window:  int   = 20,     # finestra lunga per catturare trend lenti
        s_clip:  float = 2.0,    # clipping di S_i per stabilità
    ):
        self.n_surrogates = n_surrogates
        self.W_init       = W_init
        self.T_temp       = T_temp
        self.window       = window
        self.s_clip       = s_clip

        # storia delle loss: lista di liste, una per ogni step
        # inizializzata con 1.0 per evitare divisioni strane all'inizio
        self.loss_history = [[1.0] * n_surrogates for _ in range(window)]
        self.prev_weights = [1.0] * n_surrogates

    # ── Step ────────────────────────────────────────────────────────────────

    def step(self, current_losses: List[float]) -> List[float]:
        """
        Args:
            current_losses: loss corrente per ogni surrogato (una per epoca)
        Returns:
            weights: pesi W_i aggiornati
        """
        # aggiorna la finestra: rimuovi il più vecchio, aggiungi il corrente
        self.loss_history.pop(0)
        self.loss_history.append(current_losses.copy())

        # baseline = media mobile delle ultime `window` epoche per ogni surrogato
        baseline = [
            sum(self.loss_history[t][i] for t in range(self.window)) / self.window
            for i in range(self.n_surrogates)
        ]

        # S_i = loss corrente / baseline, clippato per stabilità
        S = [
            np.clip(curr / (base + 1e-8), 1.0 / self.s_clip, self.s_clip)
            for curr, base in zip(current_losses, baseline)
        ]

        # softmax con temperatura bassa → amplifica differenze piccole
        exp_S = [np.exp(s / self.T_temp) for s in S]
        sum_e = sum(exp_S) + 1e-8
        weights = [self.W_init * (self.n_surrogates * e) / sum_e for e in exp_S]

        self.prev_weights = weights.copy()
        return weights

    # ── Reset ───────────────────────────────────────────────────────────────

    def reset(self):
        """Resetta la finestra (utile se si riparte da zero)."""
        self.loss_history = [[1.0] * self.n_surrogates for _ in range(self.window)]
        self.prev_weights = [1.0] * self.n_surrogates

    def get_weights(self) -> List[float]:
        return self.prev_weights

    # ── Stato per checkpoint ─────────────────────────────────────────────────

    def state_dict(self) -> dict:
        return {
            "loss_history": self.loss_history,
            "prev_weights": self.prev_weights,
        }

    def load_state_dict(self, state: dict) -> None:
        self.loss_history = state["loss_history"]
        self.prev_weights = state["prev_weights"]

class LossNormalizer:
    """
    Normalizza le loss dei surrogati per il loro valore alla prima epoca,
    così loss su scale diverse (es. CLIP ~0.8, VAE ~1.1) diventano
    comparabili al DynamicWeighter partendo tutte da 1.0.
    """

    def __init__(self, n_surrogates: int):
        self.n_surrogates = n_surrogates
        self.baselines    = None   # impostato alla prima chiamata

    def normalize(self, losses: List[float]) -> List[float]:
        if self.baselines is None:
            self.baselines = losses.copy()
            print(f"  LossNormalizer baselines set: {[f'{b:.4f}' for b in self.baselines]}")
        return [l / (b + 1e-8) for l, b in zip(losses, self.baselines)]

    def reset(self):
        self.baselines = None

    def state_dict(self) -> dict:
        return {"baselines": self.baselines}

    def load_state_dict(self, state: dict) -> None:
        self.baselines = state["baselines"]


# ─────────────────────────────────────────────
# 5. LOSS TOTALE
# ─────────────────────────────────────────────

def total_loss(
    # ── immagini ──
    I_im:      Tensor,
    I:         Tensor,
    M:         Tensor,
    I_target:  Tensor,
    # ── VAE ──
    posterior_im_list,  # DiagonalGaussianDistribution di I_im
    posterior_target_list,
    # ── weighter ──
    dyn_weighter: DynamicWeighter,
    # ── iperparametri ──
    alpha: float = 1.0,
    beta:  float = 1.0,
    lambda_vae:  float = 1.0,
    noise_on_mask: bool = False,
) -> Tuple[Tensor, dict]:

    # ── 1. Loss impercettibilità ──
    l_noise = noise_loss(I_im, I, M, noise_on_mask)

    # ── 2. Loss per surrogato + raccolta per il weighter ──
    per_surrogate_losses = []   # L_θi completa per ogni surrogato → S_i
    per_surrogate_terms  = []   # termini da pesare nella total loss

    for posterior_im, posterior_target in zip(posterior_im_list, posterior_target_list):
        #l_vae = vae_align_loss(posterior_im, posterior_target)
        l_vae = vae_mse(posterior_im, posterior_target)
        l_vae =  (lambda_vae *l_vae)
        per_surrogate_losses.append(l_vae.item())
        per_surrogate_terms.append(l_vae)


    # ── 3. Pesi dinamici calcolati sulla loss completa ──
    weights = dyn_weighter.get_weights()

    # ── 4. Loss surrogati pesata ──
    l_surrogates = torch.tensor(0.0, device=I.device)
    for l_i, W_i in zip(per_surrogate_terms, weights):
        l_surrogates = l_surrogates + W_i * l_i

    # ── 5. Loss totale ──
    l_tot = alpha * l_noise + beta * l_surrogates

    log = {
        "l_tot": l_tot.item(),
        "l_noise": l_noise.item(),
        "l_surrogates": l_surrogates.item(),
        "weights": weights,
    }

    # Log combined surrogate losses
    log.update({
    f"l_surrogate_{i}": l_i.item()
    for i, l_i in enumerate(per_surrogate_terms)
    })

    return l_tot, log