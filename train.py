from collections import defaultdict
import torch
import torch.optim as optim
import torch.nn.functional as F
import random
import warnings
import numpy as np
from diffusers import AutoencoderKL
from loss import total_loss, DynamicWeighter, LossNormalizer
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from data import ImmunizationDataset
from model import NestedUNet
from utils import set_seed_lib
from transformers import CLIPModel, CLIPProcessor
from tqdm import tqdm
import wandb

warnings.filterwarnings("ignore", message="QuickGELU mismatch", category=UserWarning, module="open_clip")

def save_training_checkpoint(
    checkpoint_dir, unet, optimizer, dyn_weighter, loss_normalizer,
    epoch, global_step, best_val_loss, patience_count,
    best_monitored: float = float("inf"),    # ← nuovo
):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch":           epoch,
        "global_step":     global_step,
        "best_val_loss":   best_val_loss,
        "patience_count":  patience_count,
        "best_monitored":  best_monitored,   # ← nuovo
        "unet_state":      unet.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "dyn_weighter": {
            "n_surrogates": dyn_weighter.n_surrogates,
            "W_init":       dyn_weighter.W_init,
            "T_temp":       dyn_weighter.T_temp,
            "window":       dyn_weighter.window,
            "s_clip":       dyn_weighter.s_clip,
            **dyn_weighter.state_dict(),
        },
        "loss_normalizer": loss_normalizer.state_dict(),
    }

    path = checkpoint_dir / "training_checkpoint.pth"
    torch.save(checkpoint, path)
    print(f"  ✓ Training checkpoint saved at epoch {epoch + 1} → {path}")


def load_training_checkpoint(
    checkpoint_dir, unet, optimizer, dyn_weighter, loss_normalizer, patience,
    device="cuda",
) -> dict:
    path = Path(checkpoint_dir) / "training_checkpoint.pth"

    if not path.exists():
        print("No training checkpoint found, starting from scratch.")
        return {
            "start_epoch":    0,
            "global_step":    0,
            "best_val_loss":  float("inf"),
            "patience_count": 0,
            "best_monitored": float("inf"),   # ← nuovo
        }

    checkpoint = torch.load(path, map_location=device)

    unet.load_state_dict(checkpoint["unet_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])

    dw = checkpoint["dyn_weighter"]
    dyn_weighter.n_surrogates = dw["n_surrogates"]
    dyn_weighter.W_init       = dw["W_init"]
    dyn_weighter.T_temp       = dw["T_temp"]
    dyn_weighter.window       = dw["window"]
    dyn_weighter.s_clip       = dw["s_clip"]
    dyn_weighter.load_state_dict({
        "loss_history": dw["loss_history"],
        "prev_weights": dw["prev_weights"],
    })

    if "loss_normalizer" in checkpoint:
        loss_normalizer.load_state_dict(checkpoint["loss_normalizer"])

    print(
        f"Training checkpoint loaded: "
        f"epoch={checkpoint['epoch'] + 1}, "
        f"global_step={checkpoint['global_step']}, "
        f"best_val_loss={checkpoint['best_val_loss']:.4f}, "
        f"best_monitored={checkpoint.get('best_monitored', float('inf')):.6f}"
    )

    return {
        "start_epoch":    checkpoint["epoch"] + 1,
        "global_step":    checkpoint["global_step"],
        "best_val_loss":  checkpoint["best_val_loss"],
        "patience_count": 0, # non usato
        "best_monitored": checkpoint.get("best_monitored", float("inf")),  # ← nuovo
    }



def training_loop(
        unet,
        nb_filter,
        dataloader,
        val_dataloader,
        dataset: str,
        n_epochs: int = 10,
        lr: float = 1e-4,
        batch: int = 2,
        weight_decay: float = 0.01,
        alpha: float = 1.0,
        beta: float = 1.0,
        lambda_vae: float = 0.03,
        eta: float = 0.2,
        eps: float = 32/255 * 2,
        val_every: int = 1,
        best_checkpoint_path: str = "checkpoints/unet_best.pth",
        training_checkpoint_dir: str = "checkpoints/training",
        device: str = "cuda",
        resume_from_checkpoint: bool = True,
        resume_only_weights=False,
        noise_on_mask: bool = False,
        dyn_weight_window: int   = 20,
        dyn_weight_T_temp: float = 0.1,
        dyn_weight_s_clip: float = 2.0,
):
    # ── Surrogate CLIP ──
    surrogate_clip_configs = [
        "openai/clip-vit-base-patch32",
        "openai/clip-vit-base-patch16",
        "openai/clip-vit-large-patch14"
    ]
    surrogate_clip_models = []
    for model_name in surrogate_clip_configs:
        model = CLIPModel.from_pretrained(model_name).to(device).eval()
        for param in model.parameters():
            param.requires_grad = False
        surrogate_clip_models.append(model)

    '''vae = AutoencoderKL.from_pretrained(
        "runwayml/stable-diffusion-inpainting", subfolder="vae"
    ).to(device).eval()
    for param in vae.parameters():
        param.requires_grad = False'''

    n_surrogates = len(surrogate_clip_models) #+ 1  # 3 CLIP + 1 VAE

    # ── Ottimizzatore, weighter e normalizer ──
    optimizer       = optim.AdamW(unet.parameters(), lr=lr, weight_decay=weight_decay)
    dyn_weighter    = DynamicWeighter(
        n_surrogates=n_surrogates,
        T_temp=dyn_weight_T_temp,
        window=dyn_weight_window,
        s_clip=dyn_weight_s_clip,
    )
    loss_normalizer = LossNormalizer(n_surrogates=n_surrogates)

    # ── Carica checkpoint ──
    if resume_from_checkpoint:
        state = load_training_checkpoint(
            training_checkpoint_dir, unet, optimizer,
            dyn_weighter, loss_normalizer, device,
        )
    elif resume_only_weights:
        unet.load_state_dict(torch.load(best_checkpoint_path, map_location=device, weights_only=True))
        print("resume_only_weights=True, starting fine-tuning...")
        state = {"start_epoch": 0, "global_step": 0,
                 "best_val_loss": float("inf"),
                 "best_monitored": float("inf")}
    else:
        print("resume_from_checkpoint=False, starting from scratch.")
        state = {"start_epoch": 0, "global_step": 0,
                 "best_val_loss": float("inf"),
                 "best_monitored": float("inf")}

    start_epoch    = state["start_epoch"]
    global_step    = state["global_step"]
    best_val_loss  = state["best_val_loss"]
    best_monitored = state.get("best_monitored", float("inf"))

    wandb.init(
        project="immunization",
        config={
            "dataset": dataset, "n_epochs": n_epochs, "lr": lr,
            "batch size": batch, "weight_decay": weight_decay,
            "alpha": alpha, "beta": beta, "eta": eta,
            "lambda_vae": lambda_vae, "eps": eps,
            "noise_on_mask": noise_on_mask, "nb_filter": nb_filter,
            "dyn_weight_window": dyn_weight_window,
            "dyn_weight_T_temp": dyn_weight_T_temp,
            "dyn_weight_s_clip": dyn_weight_s_clip,
            "early_stopping": "disabled",
        }
    )

    run_id = wandb.run.id
    best_checkpoint_path    = str(Path(best_checkpoint_path).parent / f"unet_best_{run_id}.pth")
    training_checkpoint_dir = str(Path(training_checkpoint_dir) / run_id)

    if start_epoch >= n_epochs:
        print(f"Training già completato ({start_epoch}/{n_epochs} epoche).")
        return unet

    unet.train()
    if not resume_from_checkpoint:
        dyn_weighter.reset()
        loss_normalizer.reset()

    for epoch in range(start_epoch, n_epochs):

        train_metrics = defaultdict(float)
        train_weights = []

        for batch_idx, (I, M, I_target) in enumerate(
                tqdm(dataloader, desc=f"Epoch {epoch + 1}/{n_epochs}", leave=False)):
            I        = I.to(device)
            M        = M.to(device)
            I_target = I_target.to(device)

            optimizer.zero_grad()
            unet_out = unet(I)
            unet_out = torch.clamp(unet_out, -eps, eps)
            if noise_on_mask:
                unet_out = unet_out * (1 - M)

            I_im = torch.clamp(I + unet_out, -1.0, 1.0)

            X_cls_list,   Y_cls_list   = [], []
            X_patch_list, Y_patch_list = [], []
            I_im_clip     = to_clip_space(I_im)
            I_target_clip = to_clip_space(I_target)

            for clip_model in surrogate_clip_models:
                X_cls, X_patch = get_visual_tokens(clip_model, I_im_clip)
                Y_cls, Y_patch = get_visual_tokens(clip_model, I_target_clip)
                X_cls_list.append(X_cls);    Y_cls_list.append(Y_cls)
                X_patch_list.append(X_patch); Y_patch_list.append(Y_patch)

            # ── VAE ──
            #posterior_im = vae.encode(I_im).latent_dist
            #posterior_target = vae.encode(I_target).latent_dist

            loss, log = total_loss(
                I_im=I_im, I=I, M=1 - M,
                X_cls_list=X_cls_list, Y_cls_list=Y_cls_list,
                X_patch_list=X_patch_list, Y_patch_list=Y_patch_list,
                posterior_im=None, posterior_target=None,
                dyn_weighter=dyn_weighter,
                alpha=alpha, beta=beta, eta=eta, lambda_vae=lambda_vae,
                noise_on_mask=noise_on_mask,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=1.0)
            optimizer.step()

            for k, v in log.items():
                if k == "weights":
                    train_weights.append(v)
                else:
                    train_metrics[k] += v
            global_step += 1

        # ── Medie training ──
        n_train = len(dataloader)
        train_metrics = {k: v / n_train for k, v in train_metrics.items()}

        n_surrogates = sum(1 for k in train_metrics if k.startswith("l_surrogate_"))
        surrogate_str = "  ".join(
            f"l_surrogate_{i}={train_metrics[f'l_surrogate_{i}']:.4f}"
            for i in range(n_surrogates)
        )

        # ── Aggiornamento DynamicWeighter ──
        current_loss_surrogates = [
            train_metrics[f"l_surrogate_{i}"] for i in range(n_surrogates)
        ]
        normalized_losses = loss_normalizer.normalize(current_loss_surrogates)
        dyn_weighter.step(normalized_losses)
        weights = dyn_weighter.get_weights()

        print(
            f"\n── Epoch {epoch + 1} Train ──  "
            f"loss={train_metrics['l_tot']:.4f}  "
            f"l_noise={train_metrics['l_noise']:.4f}  "
            f"l_surrogates={train_metrics['l_surrogates']:.4f}  "
            f"{surrogate_str}  "
            f"norm={[f'{l:.4f}' for l in normalized_losses]}  "
            f"weights={[f'{w:.4f}' for w in weights]}"
        )

        # ── Validation ──
        if (epoch + 1) % val_every == 0:
            val_metrics = validation_loop(
                unet=unet, val_dataloader=val_dataloader,
                surrogate_clip_models=surrogate_clip_models, vae=None,
                dyn_weighter=dyn_weighter, alpha=alpha, beta=beta, eta=eta,
                lambda_vae=lambda_vae, device=device, noise_on_mask=noise_on_mask,
            )

            n_surrogates_val = sum(1 for k in val_metrics if k.startswith("l_surrogate_"))
            surrogate_str_val = "  ".join(
                f"l_surrogate_{i}={val_metrics[f'l_surrogate_{i}']:.4f}"
                for i in range(n_surrogates_val)
            )

            val_surrogates = [val_metrics[f"l_surrogate_{i}"] for i in range(n_surrogates_val)]
            val_normalized = loss_normalizer.normalize(val_surrogates)
            weights_sum    = sum(weights)
            monitored      = sum(w * l for w, l in zip(weights, val_normalized)) / weights_sum

            print(
                f"── Epoch {epoch + 1} Val ──    "
                f"loss={val_metrics['l_tot']:.4f}  "
                f"l_noise={val_metrics['l_noise']:.4f}  "
                f"l_surr={val_metrics['l_surrogates']:.4f}  "
                f"{surrogate_str_val}  "
                f"monitored={monitored:.6f}\n"
            )

            # ── Best model: salva sempre il migliore senza early stopping ──
            if monitored < best_monitored:
                best_monitored = monitored
                best_val_loss  = val_metrics["l_tot"]
                Path(best_checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save(unet.state_dict(), best_checkpoint_path)
                print(
                    f"  ✓ Best model saved "
                    f"(monitored={best_monitored:.6f}, val_loss={best_val_loss:.4f})"
                    f" → {best_checkpoint_path}\n"
                )

            wandb.log({
                "train/loss":         train_metrics["l_tot"],
                "train/l_noise":      train_metrics["l_noise"],
                "train/l_surrogates": train_metrics["l_surrogates"],
                **{f"train/l_surrogate_{i}": train_metrics[f"l_surrogate_{i}"]
                   for i in range(n_surrogates)},
                **{f"train/l_surrogate_{i}_norm": normalized_losses[i]
                   for i in range(n_surrogates)},
                **{f"train/weight_{i}": w for i, w in enumerate(weights)},
                "val/loss":           val_metrics["l_tot"],
                "val/l_noise":        val_metrics["l_noise"],
                "val/l_surrogates":   val_metrics["l_surrogates"],
                **{f"val/l_surrogate_{i}": val_metrics[f"l_surrogate_{i}"]
                   for i in range(n_surrogates_val)},
                **{f"val/l_surrogate_{i}_norm": val_normalized[i]
                   for i in range(n_surrogates_val)},
                "val/monitored":      monitored,
                "epoch": epoch + 1,
            }, step=epoch + 1)

        # ── Checkpoint ──
        save_training_checkpoint(
            checkpoint_dir=training_checkpoint_dir,
            unet=unet, optimizer=optimizer,
            dyn_weighter=dyn_weighter,
            loss_normalizer=loss_normalizer,
            epoch=epoch, global_step=global_step,
            best_val_loss=best_val_loss,
            patience_count=0,              # non usato ma mantenuto per compatibilità checkpoint
            best_monitored=best_monitored,
        )

    wandb.finish()
    return unet


def to_clip_space(x: torch.Tensor) -> torch.Tensor:
    """
    Da [-1, 1] (B, 3, H, W) → normalizzazione CLIP 224x224
    """
    # [-1, 1] → [0, 1]
    x = (x + 1.0) / 2.0

    ''''# Resize a 224x224
    x = torch.nn.functional.interpolate(
        x, size=(224, 224), mode="bilinear", align_corners=False
    )'''

    # Normalizzazione CLIP
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=x.device).view(1, 3, 1, 1)
    std  = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std

def get_visual_tokens(clip_model: CLIPModel, x: torch.Tensor):
    """
    Estrae CLS token e patch tokens dal vision encoder di CLIP (HuggingFace).
    x: (B, 3, 224, 224) normalizzato per CLIP
    """
    outputs = clip_model.vision_model(pixel_values=x, output_hidden_states=False)

    # last_hidden_state: (B, N+1, D)  — CLS + patch tokens
    tokens     = outputs.last_hidden_state         # (B, N+1, D)
    cls_token  = outputs.pooler_output             # (B, D)  già estratto
    cls_token  = F.normalize(cls_token, dim=-1)
    patch_tokens = tokens[:, 1:, :]                # (B, N, D)

    return cls_token, patch_tokens




def validation_loop(
    unet,
    val_dataloader,
    surrogate_clip_models,
    vae,
    dyn_weighter,
    alpha:       float = 1.0,
    beta:        float = 1.0,
    eta:         float = 0.2,
    lambda_vae: float = 0.03,
    noise_on_mask: bool = False,
    device:      str   = "cuda",
) -> dict:

    unet.eval()
    val_metrics = {}

    with torch.no_grad():
        for I, M, I_target in tqdm(val_dataloader, desc="Validation", leave=False):
            I        = I.to(device)
            M        = M.to(device)
            I_target = I_target.to(device)

            # ── Forward ──
            unet_out = unet(I)
            if noise_on_mask:
                unet_out = unet_out * (1 - M)
            I_im     = torch.clamp(I + unet_out, -1.0, 1.0)

            # ── Feature CLIP ──
            X_cls_list,   Y_cls_list   = [], []
            X_patch_list, Y_patch_list = [], []

            I_im_clip = to_clip_space(I_im)
            I_target_clip = to_clip_space(I_target)

            for clip_model in surrogate_clip_models:
                X_cls, X_patch = get_visual_tokens(clip_model, I_im_clip)
                Y_cls, Y_patch = get_visual_tokens(clip_model, I_target_clip)
                X_cls_list.append(X_cls)
                Y_cls_list.append(Y_cls)
                X_patch_list.append(X_patch)
                Y_patch_list.append(Y_patch)

            # ── VAE ──
            #posterior_im     = vae.encode(I_im).latent_dist
            #posterior_target = vae.encode(I_target).latent_dist

            # ── Loss — total_loss gestisce pesi e dyn_weighter internamente ──
            _, log = total_loss(I_im=I_im, I=I, M=1 -M, X_cls_list=X_cls_list, Y_cls_list=Y_cls_list,
                                X_patch_list=X_patch_list, Y_patch_list=Y_patch_list, posterior_im= None,
                                posterior_target= None, dyn_weighter=dyn_weighter, alpha=alpha, beta=beta, eta=eta,
                                lambda_vae=lambda_vae, noise_on_mask=noise_on_mask)

            for k, v in log.items():
                if k == "weights":
                    continue
                val_metrics[k] = val_metrics.get(k, 0.0) + v
    # ── Medie ──
    n = len(val_dataloader)
    val_metrics = {k: v / n for k, v in val_metrics.items()}

    unet.train()
    return val_metrics





# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def seed_worker(worker_id):
    """Propaga il seed globale ai worker del DataLoader."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def print_gpu_memory(label: str = ""):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved  = torch.cuda.memory_reserved()  / 1024**3
        total     = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"[{label}] GPU Memory — allocated: {allocated:.2f}GB | reserved: {reserved:.2f}GB | total: {total:.2f}GB")

if __name__ == "__main__":

    SEED = 2023
    set_seed_lib(SEED)

    device = "cuda:1" if torch.cuda.is_available() else "cpu"

    DEBUG = False
    N_DEBUG = 100

    dataset = "DiffVax" #DiffVax | Oxford-Pet | COCO

    train_dataset = ImmunizationDataset(dataset= dataset, split="train")
    val_dataset = ImmunizationDataset(dataset= dataset, split="validation")

    if DEBUG:
        train_dataset = Subset(train_dataset, range(N_DEBUG))
        val_dataset = Subset(val_dataset, range(N_DEBUG))
    # Generator con lo stesso seed per lo shuffle
    g = torch.Generator()
    g.manual_seed(SEED)

    batch=16

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch,
        shuffle=True,
        num_workers=4,
        worker_init_fn=seed_worker,
        generator=g,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch,
        shuffle=False,
        num_workers=4,
        worker_init_fn=seed_worker,
    )

    nb_filter = [32,64,128,256,512]

    unet = NestedUNet(num_classes=3, nb_filter=nb_filter).to(device)

    trained_unet = training_loop(
        unet=unet,
        nb_filter= nb_filter,
        dataloader=train_loader,
        val_dataloader=val_loader,
        dataset= dataset,
        n_epochs=10000,
        lr=1e-4,
        batch=batch,
        weight_decay=1e-2,
        alpha=1.0,
        beta=1.0,
        eta=0.2,
        lambda_vae = 0.05,
        eps= (32 / 255 * 2),
        val_every=1,
        best_checkpoint_path="checkpoints/unet_best_nvhpvhxb.pth",
        training_checkpoint_dir="checkpoints/training/nvhpvhxb",
        device=device,
        resume_from_checkpoint=True, # Cambia a False per ricominciare da zero
        resume_only_weights = False, # True per caricare i pesi dal checkpoint
        noise_on_mask=True,
        dyn_weight_window= 30,  # ← nuovo parametro
        dyn_weight_T_temp = 0.1,  # ← nuovo parametro
        dyn_weight_s_clip= 2.0,
    )
