import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from typing import List, Tuple
from sklearn.cluster import KMeans


# ─────────────────────────────────────────────
# 1. LOSS DI IMPERCETTIBILITÀ
# ─────────────────────────────────────────────

def noise_loss(I_im: Tensor, I: Tensor, M: Tensor, noise_on_mask: bool = False) -> Tensor:
    if noise_on_mask:
        diff = (I_im - I) * M
        norm = M.sum().clamp(min=1.0)
    else:
        diff = I_im - I
        norm = torch.tensor(diff.numel(), dtype=torch.float, device=diff.device)
    return torch.abs(diff).sum() / norm


# ─────────────────────────────────────────────
# 2. LOSS SEMANTICA CLIP
# ─────────────────────────────────────────────

def coarse_loss(X: Tensor, Y: Tensor) -> Tensor:
    return (1 - F.cosine_similarity(X, Y, dim=-1)).mean()


def _kmeans_cluster(tokens: Tensor, n_clusters: int = 10) -> Tensor:
    tokens_np = tokens.detach().cpu().float().numpy()
    kmeans    = KMeans(n_clusters=n_clusters, n_init=5, random_state=0)
    kmeans.fit(tokens_np)
    return torch.tensor(kmeans.cluster_centers_, dtype=tokens.dtype, device=tokens.device)


def _sinkhorn(C: Tensor, eps: float = 0.1, n_iters: int = 50) -> Tensor:
    K_mat = torch.exp(-C / eps)
    a = torch.ones(C.shape[0], device=C.device) / C.shape[0]
    b = torch.ones(C.shape[1], device=C.device) / C.shape[1]
    u = torch.ones_like(a)
    for _ in range(n_iters):
        u = a / (K_mat @ (b / (K_mat.T @ u + 1e-8)) + 1e-8)
    v  = b / (K_mat.T @ u + 1e-8)
    pi = torch.diag(u) @ K_mat @ torch.diag(v)
    return pi


def fine_loss(X_tokens: Tensor, Y_tokens: Tensor,
              n_clusters: int = 10, sinkhorn_eps: float = 0.1,
              sinkhorn_iters: int = 50) -> Tensor:
    total = 0.0
    for b in range(X_tokens.shape[0]):
        X_clu  = _kmeans_cluster(X_tokens[b], n_clusters)
        Y_clu  = _kmeans_cluster(Y_tokens[b], n_clusters)
        X_norm = F.normalize(X_clu, dim=-1)
        Y_norm = F.normalize(Y_clu, dim=-1)
        C      = 1 - X_norm @ Y_norm.T
        pi     = _sinkhorn(C, sinkhorn_eps, sinkhorn_iters)
        total += (C * pi).sum()
    return total / X_tokens.shape[0]


# ─────────────────────────────────────────────
# 3. LOSS VAE
# ─────────────────────────────────────────────

def vae_align_loss(z_im: Tensor, z_target: Tensor) -> Tensor:
    return F.mse_loss(z_im, z_target)


def vae_pca_loss(mu: Tensor, log_var: Tensor) -> Tensor:
    kl = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
    return -kl.mean()


# ─────────────────────────────────────────────
# 4. PESATURA DINAMICA
# ─────────────────────────────────────────────

class DynamicWeighter:
    """
    Calcola i pesi adattativi W_i per l'ensemble di surrogati.

    S_i(T) = L_i(T) / L_i(T-1)
    W_i    = W_init * (t * exp(S_i / T_temp)) / sum_j(exp(S_j / T_temp))
    """

    def __init__(self, n_surrogates: int, W_init: float = 1.0, T_temp: float = 1.0):
        self.n_surrogates = n_surrogates
        self.W_init       = W_init
        self.T_temp       = T_temp
        self.prev_losses  = [1.0] * n_surrogates
        self.prev_weights = [1.0] * n_surrogates

    def step(self, current_losses: List[float]) -> List[float]:
        """
        Args:
            current_losses : loss corrente per ogni surrogato
            t              : step corrente
        Returns:
            weights : pesi W_i aggiornati
        """
        S     = [curr / (prev + 1e-8) for curr, prev in zip(current_losses, self.prev_losses)]
        exp_S = [np.exp(s / self.T_temp) for s in S]
        sum_e = sum(exp_S) + 1e-8
        weights = [self.W_init * (self.n_surrogates * e) / sum_e for e in exp_S]

        self.prev_losses = current_losses.copy()
        self.prev_weights = weights.copy()

        return weights

    def reset(self):
        """Resetta le loss precedenti (utile a inizio epoca)."""
        self.prev_losses = [1.0] * self.n_surrogates

    def get_weights(self):
        return self.prev_weights


# ─────────────────────────────────────────────
# 5. LOSS TOTALE
# ─────────────────────────────────────────────

def total_loss(
    # ── immagini ──
    I_im:      Tensor,
    I:         Tensor,
    M:         Tensor,
    # ── feature CLIP (una lista per surrogato) ──
    X_cls_list:    List[Tensor],
    Y_cls_list:    List[Tensor],
    X_patch_list:  List[Tensor],
    Y_patch_list:  List[Tensor],
    # ── VAE ──
    z_im:      Tensor,
    z_target:  Tensor,
    mu:        Tensor,
    log_var:   Tensor,
    # ── weighter ──
    dyn_weighter: DynamicWeighter,
    # ── iperparametri ──
    alpha: float = 1.0,
    beta:  float = 1.0,
    eta:   float = 0.2,
    lambda_vae: float = 0.03,
    noise_on_mask: bool = False,
) -> Tuple[Tensor, dict]:

    # ── 1. Loss impercettibilità ──
    l_noise = noise_loss(I_im, I, M, noise_on_mask)

    # ── 2. Loss per surrogato + raccolta per il weighter ──
    per_surrogate_losses = []   # L_θi completa per ogni surrogato → S_i
    per_surrogate_terms  = []   # termini da pesare nella total loss

    '''for X_cls, Y_cls, X_patch, Y_patch in zip(
        X_cls_list, Y_cls_list, X_patch_list, Y_patch_list
    ):
        l_coa_i = coarse_loss(X_cls, Y_cls)
        l_fin_i = fine_loss(X_patch, Y_patch)

        # Loss completa del surrogato i-esimo (eq. 10 paper)
        l_i = l_coa_i + eta * l_fin_i

        per_surrogate_losses.append(l_i.item())   # scalare per il weighter
        per_surrogate_terms.append(l_i)           # tensore per il backward'''

    l_vae = vae_align_loss(z_im, z_target) + vae_pca_loss(mu, log_var)
    l_vae = lambda_vae * l_vae
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
        **{f"l_surrogate_{i}": l_i.item() for i, l_i in enumerate(per_surrogate_terms)},
    }
    return l_tot, log