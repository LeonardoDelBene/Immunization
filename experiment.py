import os
os.environ["HF_HOME"] = "/equilibrium/ldelbene/cache/hf"
import csv
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from data import ImmunizationDataset
from loss import vae_mse, noise_loss
from metrics.factory import create_metric, MetricType
from model import Attack, Immunization
from utils import load_image_from_path, prepare_mask_and_masked_image, recover_image, set_seed_lib


def tensor_to_pil(img: torch.Tensor) -> Image.Image:
    img = img.detach().cpu().float()
    if img.ndim == 4 and img.shape[0] == 1:
        img = img.squeeze(0)
    img = (img / 2.0 + 0.5).clamp(0.0, 1.0)
    arr = (img.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr)


def mask_tensor_to_pil(mask: torch.Tensor) -> Image.Image:
    mask = mask.detach().cpu().float()
    while mask.ndim > 2 and mask.shape[0] == 1:
        mask = mask.squeeze(0)
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask.squeeze(0)
    arr = (mask.numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr, mode="L")


def load_experiment_sample(
    config: Dict[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor, Image.Image, Image.Image]:
    if config.get("input_image_path") and config.get("input_mask_path"):
        image = load_image_from_path(config["input_image_path"], size=(512, 512))
        mask = load_image_from_path(config["input_mask_path"], size=(512, 512)).convert("L")
        mask_t, _, image_t = prepare_mask_and_masked_image(image, mask)
        return image_t, mask_t, image, mask

    dataset = ImmunizationDataset(
        dataset=config.get("dataset_type", "DiffVax"),
        split=config.get("dataset_split", "validation"),
        image_size=config.get("image_size", 512),
    )
    sample_idx = config.get("sample_idx", 0)
    if sample_idx >= len(dataset):
        raise IndexError(f"sample_idx {sample_idx} out of range for dataset of size {len(dataset)}")

    image_t, mask_t = dataset[sample_idx]
    image_t = image_t.unsqueeze(0)
    mask_t = mask_t.unsqueeze(0)

    image_pil = tensor_to_pil(image_t)
    mask_pil = mask_tensor_to_pil(mask_t)
    return image_t, mask_t, image_pil, mask_pil


def load_models(config: Dict[str, Any], device: torch.device):
    attack_model = Attack()
    immunization_model = Immunization(
        device=str(device),
        load_existing=config.get("load_existing", True),
        load_path=config.get("checkpoint_path"),
        vae=attack_model.model.vae,
        molt_filter=config.get("molt_filter", 1),
    )
    return immunization_model


# ─────────────────────────────────────────────────────────────────────────────
# Core helpers
# ─────────────────────────────────────────────────────────────────────────────

def _immunize_and_score(
    immunization_model: Immunization,
    image_tensor: torch.Tensor,
    mask_tensor: torch.Tensor,
    image_pil: Image.Image,
    mask_pil: Image.Image,
    immunize_kwargs: Dict[str, Any],
    variable_name: str,
    param_value: float,
    masked_metric,
) -> Dict[str, float]:
    """Immunize a single image with the given kwargs and return all metrics."""
    img = image_tensor.to(immunization_model.device).float()
    mask = mask_tensor.to(immunization_model.device).float()

    img_final, l_vae, l_noise = immunization_model.immunize_img_targeted(img=img, img_mask=mask, **immunize_kwargs)

    img_final_pil = tensor_to_pil(img_final)
    img_final_recovered = recover_image(img_final_pil, image_pil, mask_pil, background=True)

    lpips_scores = masked_metric.compute(image_pil, img_final_recovered, mask_pil)

    return {
        variable_name: param_value,
        "l_vae": l_vae,
        "l_noise": l_noise,
        "subject_lpips": lpips_scores["subject_lpips"],
        "global_lpips": lpips_scores["global_lpips"],
    }


def _base_kwargs(config: Dict[str, Any]) -> Dict[str, Any]:
    """Fixed immunization kwargs shared across all sweeps."""
    return {
        "noise_mode": config.get("noise_mode", "mask"),
        "is_2_stage": config.get("is_2_stage", True),
        "pgd": config.get("pgd", True),
        "lr": config.get("lr", 1e-4),
        "lambda_vae": config.get("lambda_vae", 1.0),
        "n_steps": config.get("n_steps", 50),
        "eps": config.get("eps", 8 / 255.0),
        "lambda_noise": config.get("lambda_noise", 100.0),
    }


def _save_csv(results: List[Dict[str, float]], output_dir: str, filename: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, filename)
    with open(csv_path, mode="w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved CSV: {csv_path}")

def _plot_results(
    results: List[Dict[str, float]],
    variable_name: str,
    output_dir: str,
    experiment_name: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    xs = [r[variable_name] for r in results]

    # -------------------------
    # VAE LOSS
    # -------------------------
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        xs,
        [r["l_vae"] for r in results],
        marker="o",
    )

    ax.set_title(f"{experiment_name} - VAE Loss")
    ax.set_xlabel(variable_name)
    ax.set_ylabel("VAE Loss")
    ax.grid(True, linestyle="--", alpha=0.4)

    path = os.path.join(
        output_dir,
        f"vae_loss_{experiment_name}.png",
    )
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {path}")

    # -------------------------
    # NOISE LOSS
    # -------------------------
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        xs,
        [r["l_noise"] for r in results],
        marker="s",
    )

    ax.set_title(f"{experiment_name} - Noise Loss")
    ax.set_xlabel(variable_name)
    ax.set_ylabel("Noise Loss")
    ax.grid(True, linestyle="--", alpha=0.4)

    path = os.path.join(
        output_dir,
        f"noise_loss_{experiment_name}.png",
    )
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {path}")

    # -------------------------
    # LPIPS
    # -------------------------
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        xs,
        [r["subject_lpips"] for r in results],
        marker="o",
        label="Subject LPIPS",
    )

    ax.plot(
        xs,
        [r["global_lpips"] for r in results],
        marker="s",
        label="Global LPIPS",
    )

    ax.set_title(f"{experiment_name} - LPIPS")
    ax.set_xlabel(variable_name)
    ax.set_ylabel("LPIPS")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()

    path = os.path.join(
        output_dir,
        f"lpips_{experiment_name}.png",
    )
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Per-parameter experiment functions
# ─────────────────────────────────────────────────────────────────────────────

def experiment_n_steps(
    immunization_model: Immunization,
    image_tensor: torch.Tensor,
    mask_tensor: torch.Tensor,
    image_pil: Image.Image,
    mask_pil: Image.Image,
    masked_metric,
    config: Dict[str, Any],
    values: Optional[List[int]] = None,
) -> List[Dict[str, float]]:
    """Sweep over n_steps, keeping eps and lambda_noise fixed."""
    values = values or config.get("sweep_n_steps", [10, 20, 50, 100])
    results = []
    for v in values:
        print(f"[n_steps] {v}")
        kwargs = {**_base_kwargs(config), "n_steps": v}
        results.append(
            _immunize_and_score(
                immunization_model,
                image_tensor,
                mask_tensor,
                image_pil,
                mask_pil,
                kwargs,
                "n_steps",
                v,
                masked_metric,
            )
        )
    _save_csv(results, config["output_dir"], "sweep_n_steps.csv")
    _plot_results(results, "n_steps", config["output_dir"],experiment_name="n_steps")
    return results


def experiment_eps(
    immunization_model: Immunization,
    image_tensor: torch.Tensor,
    mask_tensor: torch.Tensor,
    image_pil: Image.Image,
    mask_pil: Image.Image,
    masked_metric,
    config: Dict[str, Any],
    values: Optional[List[float]] = None,
) -> List[Dict[str, float]]:
    """Sweep over eps, keeping n_steps and lambda_noise fixed."""
    values = values or config.get("sweep_eps", [8 / 255.0, 16 / 255.0, 32 / 255.0, 64 / 255.0])
    results = []
    for v in values:
        print(f"[eps] {v:.5f}")
        kwargs = {**_base_kwargs(config), "eps": v}
        results.append(
            _immunize_and_score(
                immunization_model,
                image_tensor,
                mask_tensor,
                image_pil,
                mask_pil,
                kwargs,
                "eps",
                v,
                masked_metric,
            )
        )
    _save_csv(results, config["output_dir"], "sweep_eps.csv")
    _plot_results(results, "eps", config["output_dir"], experiment_name="eps")
    return results


def experiment_lambda_noise(
    immunization_model: Immunization,
    image_tensor: torch.Tensor,
    mask_tensor: torch.Tensor,
    image_pil: Image.Image,
    mask_pil: Image.Image,
    masked_metric,
    config: Dict[str, Any],
    values: Optional[List[float]] = None,
) -> List[Dict[str, float]]:
    """Sweep over lambda_noise, keeping n_steps and eps fixed."""
    values = values or config.get("sweep_lambda_noise", [10.0, 50.0, 100.0, 200.0])
    results = []
    for v in values:
        print(f"[lambda_noise] {v}")
        kwargs = {**_base_kwargs(config), "lambda_noise": v}
        results.append(
            _immunize_and_score(
                immunization_model,
                image_tensor,
                mask_tensor,
                image_pil,
                mask_pil,
                kwargs,
                "lambda_noise",
                v,
                masked_metric,
            )
        )
    _save_csv(results, config["output_dir"], "sweep_lambda_noise.csv")
    _plot_results(results, "lambda_noise", config["output_dir"],"lambda_noise")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_experiments(config: Dict[str, Any]) -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    set_seed_lib(config.get("seed", 42))

    immunization_model = load_models(config, device)
    image_tensor, mask_tensor, image_pil, mask_pil = load_experiment_sample(config)
    masked_metric = create_metric(MetricType.MASKED, lpips_net="alex")

    os.makedirs(config["output_dir"], exist_ok=True)

    shared = (immunization_model, image_tensor, mask_tensor, image_pil, mask_pil, masked_metric, config)

    print("\n" + "=" * 60)
    print("Experiment 1/3 — n_steps sweep")
    #experiment_n_steps(*shared)

    print("\n" + "=" * 60)
    print("Experiment 2/3 — eps sweep")
    experiment_eps(*shared)

    print("\n" + "=" * 60)
    print("Experiment 3/3 — lambda_noise sweep")
    #experiment_lambda_noise(*shared)

    print("\nAll experiments completed.")


def get_default_config() -> Dict[str, Any]:
    return {
        "dataset_type": "DiffVax",
        "dataset_split": "validation",
        "sample_idx": 0,
        "image_size": 512,
        "noise_mode": "all",
        "is_2_stage": True,
        "pgd": False,
        "lr": 1e-4,
        "lambda_vae": 1.0,
        # fixed defaults used when a parameter is NOT being swept
        "n_steps": 300, #valore standard 300
        "eps": 32 / 255.0,
        "lambda_noise": 100.0, #valore standard 100
        "load_existing": True,
        "checkpoint_path": os.path.join("checkpoints", "unet_best_fk2utznx.pth"),
        "molt_filter": 2,
        "seed": 2043,
        "local_files_only": True,
        "output_dir": os.path.join("experiment", "unet_best_fk2utznx"),
        # sweep grids
        "sweep_n_steps": [50, 100, 200, 300,400,500,600,700,800,900,1000],
        "sweep_eps": [8 / 255.0, 16 / 255.0, 32 / 255.0, 64 / 255.0, 128/255 ],
        "sweep_lambda_noise": [1.0, 10.0, 50.0, 100.0, 150.0, 200.0],
    }


if __name__ == "__main__":
    config = get_default_config()
    run_experiments(config)