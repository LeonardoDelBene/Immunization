"""
Diagnostico: la delta prodotta da targeted_unet_refinement e' overfit
alla geometria locale del latent space del VAE, o e' una direzione
ragionevolmente "stabile"?

Metrica primaria: l_vae, la stessa loss usata in targeted_unet_refinement
(MSE tra posterior.mean di img_final e posterior_target.mean). Tutti i
test sono valutati in termini di questa loss, perche' e' quella che stai
davvero minimizzando durante l'attacco -- non un proxy.

Tre test, tutti eseguiti SUL DELTA GIA' OTTIMIZZATO (post-hoc, non durante
il training):

1. random_mask_sensitivity (pixel-level):
   applica solo una frazione random del delta (es. 50%) e guarda quanto
   risale l_vae rispetto al delta completo.
   Se con il 50% del delta l_vae torna quasi al livello di "nessun delta"
   -> overfit locale, il delta dipende da una combinazione molto
   specifica di pixel.
   Se l_vae degrada in modo ~lineare con la frazione mascherata -> il
   delta e' "diffuso" e robusto, niente di patologico.

2. patch_mask_sensitivity:
   stessa idea ma con maschere a blocchi spaziali (patch) invece di
   pixel sparsi random. Il VAE ha receptive field locale (conv), quindi
   mascherare pixel isolati puo' essere poco informativo: la patch mask
   e' il test piu' rilevante per la tua pipeline.

3. spatial_transform_sensitivity:
   applica piccole traslazioni / crop+resize / jitter di luminosita'
   a img_final e guarda quanto risale l_vae.
   Se basta una traslazione di pochi pixel per far risalire l_vae verso
   il livello "nessun delta" -> il delta sfrutta una scorciatoia molto
   fragile dell'encoder.

Per ogni test viene anche riportato un baseline "zero delta" (img_orig
senza alcuna perturbazione): e' il punto di riferimento per dire se un
dato degrado e' "quasi totale" o solo parziale.

Output: un dict di metriche + un plot riassuntivo (PNG). Nessuna
decisione automatica: i numeri vanno letti, non c'e' soglia magica.
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------
# Metrica primaria: l_vae (stessa identica definizione del training loop)
# ---------------------------------------------------------------------

def _vae_dtype_device(vae):
    """Dtype e device dei parametri del VAE (es. fp16 su cuda)."""
    p = next(vae.parameters())
    return p.dtype, p.device


def _match_vae(vae, *tensors):
    """
    Casta tutti i tensori passati al dtype/device del VAE.
    Necessario perche' il VAE puo' essere in fp16: se gli passiamo un
    tensore fp32 (es. da torch.ones_like su un img.float()), conv2d
    lancia 'Input type (Half) and bias type (float) should be the same'
    (o viceversa).
    """
    dtype, device = _vae_dtype_device(vae)
    return [t.to(device=device, dtype=dtype) for t in tensors]


@torch.no_grad()
def compute_l_vae(vae, img: torch.Tensor, target_mean: torch.Tensor) -> float:
    """
    MSE tra posterior.mean(img) e target_mean.
    Identica a vae_mse usata in targeted_unet_refinement, cosi' i numeri
    di questo diagnostico sono direttamente comparabili ai log di training.
    """
    img, target_mean = _match_vae(vae, img, target_mean)
    posterior = vae.encode(img).latent_dist
    # mse in fp32 per stabilita' numerica, indipendentemente dal dtype del VAE
    return F.mse_loss(posterior.mean.float(), target_mean.float()).item()


@torch.no_grad()
def compute_cosine_dist(vae, img: torch.Tensor, target_mean: torch.Tensor) -> float:
    """Solo per riferimento secondario nei print, non guida le decisioni."""
    img, target_mean = _match_vae(vae, img, target_mean)
    posterior = vae.encode(img).latent_dist
    cos = F.cosine_similarity(
        posterior.mean.float().flatten(1), target_mean.float().flatten(1), dim=-1
    ).mean().item()
    return 1.0 - cos


def make_random_pixel_mask(ref_tensor: torch.Tensor, keep_frac: float):
    """
    Maschera binaria random a livello di pixel (broadcast su canali).
    Dtype/device presi da ref_tensor, cosi' la mask e' sempre compatibile
    col tensore che andrai a moltiplicare (es. delta_total).
    """
    b, c, h, w = ref_tensor.shape
    m = (torch.rand(b, 1, h, w, device=ref_tensor.device) < keep_frac)
    m = m.to(ref_tensor.dtype)
    return m.expand(b, c, h, w)


def make_random_patch_mask(ref_tensor: torch.Tensor, keep_frac: float, patch_size: int):
    """Maschera binaria a blocchi spaziali patch_size x patch_size, dtype-aware."""
    b, c, h, w = ref_tensor.shape
    ph, pw = h // patch_size, w // patch_size
    patch_mask = (torch.rand(b, 1, ph, pw, device=ref_tensor.device) < keep_frac).float()
    m = F.interpolate(patch_mask, size=(h, w), mode="nearest")
    m = m.to(ref_tensor.dtype)
    return m.expand(b, c, h, w)


# ---------------------------------------------------------------------
# Test 1 & 2: mask sensitivity (pixel-level e patch-level), su l_vae
# ---------------------------------------------------------------------

def mask_sensitivity_sweep(
    vae,
    img_orig: torch.Tensor,
    delta_total: torch.Tensor,
    target_mean: torch.Tensor,
    keep_fracs=(1.0, 0.75, 0.5, 0.25, 0.1, 0.0),
    mode: str = "pixel",       # "pixel" o "patch"
    patch_size: int = 16,
    n_trials: int = 5,         # ripetizioni per ogni keep_frac (la mask e' random)
):
    """
    Per ogni keep_frac, applica n_trials maschere random indipendenti al
    delta_total, ricostruisce l'immagine, e misura l_vae rispetto al
    target. Ritorna media e std per ogni keep_frac.

    keep_frac=1.0 -> delta intero (baseline "attacco riuscito")
    keep_frac=0.0 -> nessun delta, equivalente a img_orig pura
                     (baseline "nessun attacco", utile come riferimento
                     per capire quanto e' "quasi totale" un degrado)

    Dtype/device sono presi dal VAE (vedi _match_vae) cosi' tutto e'
    coerente anche se il VAE e' in fp16.
    """
    img_orig, delta_total = _match_vae(vae, img_orig, delta_total)

    results = {"keep_frac": [], "l_vae_mean": [], "l_vae_std": [],
               "cos_dist_mean": [], "cos_dist_std": []}

    for kf in keep_fracs:
        l_vae_vals, cos_vals = [], []
        # a kf=0.0 o 1.0 la mask e' deterministica, non serve ripetere
        n_t = 1 if kf in (0.0, 1.0) else n_trials

        for _ in range(n_t):
            if kf == 1.0:
                mask = torch.ones_like(delta_total)
            elif kf == 0.0:
                mask = torch.zeros_like(delta_total)
            elif mode == "pixel":
                mask = make_random_pixel_mask(delta_total, kf)
            elif mode == "patch":
                mask = make_random_patch_mask(delta_total, kf, patch_size)
            else:
                raise ValueError("mode deve essere 'pixel' o 'patch'")

            img_masked = torch.clamp(img_orig + delta_total * mask, -1.0, 1.0)

            l_vae_vals.append(compute_l_vae(vae, img_masked, target_mean))
            cos_vals.append(compute_cosine_dist(vae, img_masked, target_mean))

        results["keep_frac"].append(kf)
        results["l_vae_mean"].append(float(np.mean(l_vae_vals)))
        results["l_vae_std"].append(float(np.std(l_vae_vals)))
        results["cos_dist_mean"].append(float(np.mean(cos_vals)))
        results["cos_dist_std"].append(float(np.std(cos_vals)))

    return results


# ---------------------------------------------------------------------
# Test 3: spatial transform sensitivity, su l_vae
# ---------------------------------------------------------------------

def translate_image(img: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
    """Trasla l'immagine di (dx, dy) pixel (wrap-around, va bene per shift piccoli)."""
    return torch.roll(img, shifts=(dy, dx), dims=(2, 3))


def crop_resize_image(img: torch.Tensor, crop_frac: float) -> torch.Tensor:
    """Crop centrale di crop_frac (es 0.95) e resize back alla size originale."""
    b, c, h, w = img.shape
    ch, cw = int(h * crop_frac), int(w * crop_frac)
    top, left = (h - ch) // 2, (w - cw) // 2
    cropped = img[:, :, top:top + ch, left:left + cw]
    return F.interpolate(cropped, size=(h, w), mode="bilinear", align_corners=False)


def brightness_jitter(img: torch.Tensor, delta_b: float) -> torch.Tensor:
    """Shift di luminosita' costante, clampato al range valido [-1, 1]."""
    return torch.clamp(img + delta_b, -1.0, 1.0)


def spatial_transform_sensitivity(
    vae,
    img_final: torch.Tensor,   # img_orig + delta_total, GIA' clampata
    img_orig: torch.Tensor,    # serve per il baseline "zero delta" trasformato
    target_mean: torch.Tensor,
    translations=(0, 1, 2, 4, 8),
    crop_fracs=(1.0, 0.99, 0.97, 0.95, 0.90),
    brightness_deltas=(0.0, 0.01, 0.02, 0.05),
):
    """
    Misura l_vae dopo aver applicato piccole trasformazioni geometriche o
    fotometriche a img_final. Per ogni trasformazione viene calcolato
    anche l_vae applicando la STESSA trasformazione a img_orig (senza
    delta), cosi' hai un riferimento di quanto "sale" l_vae per il solo
    effetto della trasformazione, indipendentemente dal delta.
    """
    img_final, img_orig = _match_vae(vae, img_final, img_orig)

    out = {
        "translation": {"shift_px": [], "l_vae": [], "l_vae_orig": [], "cos_dist": []},
        "crop_resize": {"crop_frac": [], "l_vae": [], "l_vae_orig": [], "cos_dist": []},
        "brightness":  {"delta_b": [], "l_vae": [], "l_vae_orig": [], "cos_dist": []},
    }

    for s in translations:
        img_t = translate_image(img_final, dx=s, dy=s)
        img_t_orig = translate_image(img_orig, dx=s, dy=s)
        out["translation"]["shift_px"].append(s)
        out["translation"]["l_vae"].append(compute_l_vae(vae, img_t, target_mean))
        out["translation"]["l_vae_orig"].append(compute_l_vae(vae, img_t_orig, target_mean))
        out["translation"]["cos_dist"].append(compute_cosine_dist(vae, img_t, target_mean))

    for cf in crop_fracs:
        img_c = crop_resize_image(img_final, cf)
        img_c_orig = crop_resize_image(img_orig, cf)
        out["crop_resize"]["crop_frac"].append(cf)
        out["crop_resize"]["l_vae"].append(compute_l_vae(vae, img_c, target_mean))
        out["crop_resize"]["l_vae_orig"].append(compute_l_vae(vae, img_c_orig, target_mean))
        out["crop_resize"]["cos_dist"].append(compute_cosine_dist(vae, img_c, target_mean))

    for db in brightness_deltas:
        img_b = brightness_jitter(img_final, db)
        img_b_orig = brightness_jitter(img_orig, db)
        out["brightness"]["delta_b"].append(db)
        out["brightness"]["l_vae"].append(compute_l_vae(vae, img_b, target_mean))
        out["brightness"]["l_vae_orig"].append(compute_l_vae(vae, img_b_orig, target_mean))
        out["brightness"]["cos_dist"].append(compute_cosine_dist(vae, img_b, target_mean))

    return out


# ---------------------------------------------------------------------
# Plot riassuntivo
# ---------------------------------------------------------------------

def plot_diagnostics(mask_pixel_res, mask_patch_res, transform_res, out_path="diagnostics.png"):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # (1) mask sensitivity su l_vae, pixel vs patch
    ax = axes[0, 0]
    ax.errorbar(mask_pixel_res["keep_frac"], mask_pixel_res["l_vae_mean"],
                yerr=mask_pixel_res["l_vae_std"], marker="o", label="pixel mask")
    ax.errorbar(mask_patch_res["keep_frac"], mask_patch_res["l_vae_mean"],
                yerr=mask_patch_res["l_vae_std"], marker="s", label="patch mask")
    ax.set_xlabel("keep_frac (frazione di delta applicata)")
    ax.set_ylabel("l_vae (mse latente)")
    ax.set_title("Mask sensitivity: l_vae")
    ax.invert_xaxis()
    ax.legend()
    ax.grid(alpha=0.3)

    # (2) stesso plot ma in log-scale, utile se l_vae varia di ordini di
    # grandezza tra keep_frac=1.0 e keep_frac=0.0
    ax = axes[0, 1]
    ax.errorbar(mask_pixel_res["keep_frac"], mask_pixel_res["l_vae_mean"],
                yerr=mask_pixel_res["l_vae_std"], marker="o", label="pixel mask")
    ax.errorbar(mask_patch_res["keep_frac"], mask_patch_res["l_vae_mean"],
                yerr=mask_patch_res["l_vae_std"], marker="s", label="patch mask")
    ax.set_yscale("log")
    ax.set_xlabel("keep_frac")
    ax.set_ylabel("l_vae (log scale)")
    ax.set_title("Mask sensitivity: l_vae (log scale)")
    ax.invert_xaxis()
    ax.legend()
    ax.grid(alpha=0.3, which="both")

    # (3) translation sensitivity su l_vae, con baseline "img_orig trasformata"
    ax = axes[1, 0]
    ax.plot(transform_res["translation"]["shift_px"], transform_res["translation"]["l_vae"],
            marker="o", label="img_final (con delta)")
    ax.plot(transform_res["translation"]["shift_px"], transform_res["translation"]["l_vae_orig"],
            marker="x", linestyle="--", color="gray", label="img_orig (no delta)")
    ax.set_xlabel("shift (px)")
    ax.set_ylabel("l_vae")
    ax.set_title("Translation sensitivity: l_vae")
    ax.legend()
    ax.grid(alpha=0.3)

    # (4) crop sensitivity su l_vae
    ax = axes[1, 1]
    crop_x = [int((1 - cf) * 100) for cf in transform_res["crop_resize"]["crop_frac"]]
    ax.plot(crop_x, transform_res["crop_resize"]["l_vae"],
            marker="o", label="img_final (con delta)")
    ax.plot(crop_x, transform_res["crop_resize"]["l_vae_orig"],
            marker="x", linestyle="--", color="gray", label="img_orig (no delta)")
    ax.set_xlabel("crop removed (%)")
    ax.set_ylabel("l_vae")
    ax.set_title("Crop+resize sensitivity: l_vae")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"[diagnostics] plot salvato in {out_path}")


# ---------------------------------------------------------------------
# Entry point di esempio
# ---------------------------------------------------------------------

def run_full_diagnostic(
    vae,
    img_orig: torch.Tensor,
    img_final: torch.Tensor,      # output di targeted_unet_refinement
    target_mean: torch.Tensor,    # self.posterior_target.mean
    out_png="diagnostics.png",
):
    """
    Esegue tutti e tre i test e stampa un riepilogo testuale, con l_vae
    come metrica di riferimento (la stessa minimizzata in training).

    Dtype/device sono presi automaticamente dal VAE (vedi _match_vae),
    quindi non serve passarli: funziona sia che il VAE sia in fp16 sia
    in fp32, indipendentemente dal dtype di img_orig/img_final in input.

    Da chiamare DOPO aver girato targeted_unet_refinement, passando
    img_final (il suo output) e img_orig (lo stesso passato alla funzione).
    """
    delta_total = (img_final - img_orig).detach()

    # baseline "zero delta": quanto vale l_vae su img_orig pura
    l_vae_zero_delta = compute_l_vae(vae, img_orig, target_mean)

    print("\n[1/3] Random PIXEL mask sensitivity...")
    mask_pixel_res = mask_sensitivity_sweep(
        vae, img_orig, delta_total, target_mean, mode="pixel"
    )

    print("[2/3] Random PATCH mask sensitivity...")
    mask_patch_res = mask_sensitivity_sweep(
        vae, img_orig, delta_total, target_mean, mode="patch", patch_size=16
    )

    print("[3/3] Spatial transform sensitivity...")
    transform_res = spatial_transform_sensitivity(vae, img_final, img_orig, target_mean)

    plot_diagnostics(mask_pixel_res, mask_patch_res, transform_res, out_path=out_png)

    # ---- Riepilogo testuale, basato su l_vae ----
    baseline_l_vae = mask_pixel_res["l_vae_mean"][mask_pixel_res["keep_frac"].index(1.0)]
    half_l_vae_pixel = mask_pixel_res["l_vae_mean"][mask_pixel_res["keep_frac"].index(0.5)]
    half_l_vae_patch = mask_patch_res["l_vae_mean"][mask_patch_res["keep_frac"].index(0.5)]

    # quanto del "miglioramento totale" (da zero-delta a delta completo) resta
    # con solo il 50% del delta -- 1.0 = nessuna perdita, 0.0 = perdita totale
    total_gap = l_vae_zero_delta - baseline_l_vae
    if abs(total_gap) > 1e-12:
        retained_pixel = 1.0 - (half_l_vae_pixel - baseline_l_vae) / total_gap
        retained_patch = 1.0 - (half_l_vae_patch - baseline_l_vae) / total_gap
    else:
        retained_pixel = retained_patch = float("nan")

    print("\n=== Riepilogo (metrica: l_vae) ===")
    print(f"l_vae con zero delta (img_orig pura):            {l_vae_zero_delta:.6f}")
    print(f"l_vae con delta completo (keep_frac=1.0):        {baseline_l_vae:.6f}")
    print(f"l_vae con 50% del delta (pixel mask):            {half_l_vae_pixel:.6f}  "
          f"(efficacia residua: {retained_pixel*100:.1f}%)")
    print(f"l_vae con 50% del delta (patch mask):            {half_l_vae_patch:.6f}  "
          f"(efficacia residua: {retained_patch*100:.1f}%)")
    print(
        "-> 'efficacia residua' ~100% significa che meta' del delta produce "
        "quasi lo stesso effetto del delta intero (delta diffuso, NON overfit "
        "locale). Efficacia residua vicina a 0% significa che il 50% random "
        "del delta non serve quasi a niente -> il delta dipende da una "
        "combinazione molto specifica di pixel/patch (overfit locale, "
        "LPAA-style motivato)."
    )

    shift2_l_vae = transform_res["translation"]["l_vae"][
        transform_res["translation"]["shift_px"].index(2)
    ]
    shift2_l_vae_orig = transform_res["translation"]["l_vae_orig"][
        transform_res["translation"]["shift_px"].index(2)
    ]
    print(f"\nl_vae dopo traslazione di 2px (img_final):       {shift2_l_vae:.6f}")
    print(f"l_vae dopo traslazione di 2px (img_orig, no delta): {shift2_l_vae_orig:.6f}")
    print(
        "-> se shift2_l_vae si avvicina a shift2_l_vae_orig (il livello "
        "'come se il delta non ci fosse'), il delta e' spazialmente fragile: "
        "bastano 2px di shift per annullarne l'effetto."
    )

    return {
        "l_vae_zero_delta": l_vae_zero_delta,
        "mask_pixel": mask_pixel_res,
        "mask_patch": mask_patch_res,
        "transform": transform_res,
    }