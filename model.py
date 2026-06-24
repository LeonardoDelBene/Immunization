from torch import nn
from PIL import Image
import torch
import lpips
from diffusers import (
    StableDiffusionInpaintPipeline,
    AutoPipelineForImage2Image,
    DDIMScheduler
)
import torch.nn.functional as F
from typing import Union, List, Optional, Callable
from diffusers import StableDiffusionInstructPix2PixPipeline
import torchvision.transforms as transforms
from loss import vae_mse, noise_loss, vae_align_loss, vae_divergence_loss
from transformers import CLIPTokenizer, CLIPTextModel
import copy
import torch.distributions as D
from diffusers import AutoencoderKL



def load_clip_vit_l_14(device: str = "cuda:0"):
    """Load OpenAI CLIP ViT-L/14 for text embeddings and tokenization."""
    model_name = "openai/clip-vit-large-patch14"
    tokenizer = CLIPTokenizer.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(device).eval()
    return model, tokenizer


class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, num_groups=8):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)  # fix: num_features + 2d
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)     # fix: typo + num_features + 2d


    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out


class UNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv2_2 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_3 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_4 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = self.final(x0_4)
        return output


class NestedUNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, nb_filter=None, **kwargs):
        super().__init__()

        if nb_filter is None:
            nb_filter = [32, 64, 128, 256, 512]

        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])

        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            output = self.final(x0_4)
            return output


class Immunization:
    def __init__(
        self,
        device:         str   = "cuda:0",
        clamp_min:      float = -1.0,
        clamp_max:      float =  1.0,
        load_existing:  bool  = False,
        load_path:      str   = None,
        vae                  = None,
        molt_filter:    int = 1,
    ):
        self.device           = device
        self.clamp_min        = clamp_min
        self.clamp_max        = clamp_max
        vae = AutoencoderKL.from_pretrained(
            "runwayml/stable-diffusion-inpainting", subfolder="vae"
        ).to(self.device).eval()
        for param in vae.parameters():
            param.requires_grad = False
        self.vae              = vae
        self.lpips_fn         = lpips.LPIPS(net="alex").to(device).eval()
        self.nb_filters      = [x * molt_filter for x in [32, 64, 128, 256, 512]]
        with torch.no_grad():
         target = Image.open("./data/gray.png").convert("RGB").resize((512, 512))
         target = transforms.ToTensor()(target)           # [0, 1]
         target = (target * 2.0 - 1.0)                   # [-1, 1]
         target = target.unsqueeze(0).to(device)
         self.posterior_target = vae.encode(target).latent_dist

        self.model = NestedUNet(num_classes=3, nb_filter=self.nb_filters).to(device)

        if load_existing:
            if load_path is None:
                raise ValueError("load_path must be specified when load_existing=True")
            self.model.load_state_dict(torch.load(load_path, weights_only=True, map_location=device))
            print(f"Checkpoint loaded from {load_path}")

        self.model.eval()

    def immunize_img(self, img: torch.Tensor, img_mask: torch.Tensor, noise="mask") -> torch.Tensor:
        img_f  = img.float().to(self.device)
        mask_f = img_mask.float().to(self.device)

        with torch.no_grad():
            unet_out = self.model(img_f)
            if noise =="mask":
                unet_out = unet_out * (1 - mask_f)

        img_adv = torch.clamp(img_f + unet_out, self.clamp_min, self.clamp_max)
        return img_adv
    

    def targeted_unet_refinement(
     self,
     img:            torch.Tensor,   # img_adv (output stage 1)
     img_orig:       torch.Tensor,   # img originale (per vincolo eps)
     img_mask:       torch.Tensor,
     noise_mode:          str   = "mask",
     eps:            float = 16/255,
     lr:             float = 1e-5,
     n_steps:        int   = 100,
     lambda_vae:     float = 1.0,
     lambda_noise:   float = 1.0,
     log_every:      int   = 10,
     # --- LPAA patch-based: regolarizzazione anti-overfit locale ---
     use_lpaa:       bool = False,
     lpaa_n_masks:   int   = 0,      # 0 = disabilitato, comportamento originale
     lpaa_patch_size: int  = 16,
     lpaa_keep_frac: float = 0.5,
    ) -> torch.Tensor:

     assert self.vae is not None, "vae must be set in __init__"
     assert self.posterior_target is not None, "posterior_target must be set"

     self.vae.eval()
     local_unet = copy.deepcopy(self.model).to(self.device)
     local_unet.eval()

     optimizer = torch.optim.Adam(local_unet.parameters(), lr=lr)

     img_adv  = img.float().to(self.device)
     img_orig = img_orig.float().to(self.device)   # riferimento per eps
     mask     = img_mask.float().to(self.device)

     
     if use_lpaa:
         from loss import lpaa_patch_loss

     for step in range(n_steps):
        
        noise    = local_unet(img_adv)
        if noise_mode == "mask":
            noise    = noise * (1 - mask)

        noise    = torch.clamp(noise, -eps, eps)

        img_final = img_adv + noise

        # vincolo L∞ rispetto all'originale
        total_delta = img_final - img_orig
        total_delta = torch.clamp(total_delta, -eps, eps)
        img_final   = torch.clamp(img_orig + total_delta, -1.0, 1.0)

        # ricalcoliamo il delta DOPO il clamp finale, e' quello che
        # verra' effettivamente mascherato nel branch LPAA
        delta_total = img_final - img_orig

        posterior_im = self.vae.encode(img_final).latent_dist
        cosine_dist =1 - F.cosine_similarity(
            posterior_im.mean, self.posterior_target.mean, dim=-1
        ).mean().item()

        # l_vae_full: loss sull'immagine intera, SEMPRE calcolata (anche
        # con LPAA attivo) solo per logging/confronto con run precedenti.
        # Non viene quasi mai usata per il gradiente quando use_lpaa=True.
        #l_vae_full = vae_align_loss(posterior_im, self.posterior_target)
        l_vae_full = vae_mse(posterior_im, self.posterior_target)

        if use_lpaa:
            # loss "vera" per il gradiente: media su N patch-mask random
            # del delta. Forza local_unet a produrre una perturbazione
            # la cui efficacia non dipenda da una combinazione molto
            # specifica di pixel/patch (vedi diagnose_local_overfit.py).
            l_vae = lpaa_patch_loss(
                vae=self.vae,
                img_orig=img_orig,
                delta_total=delta_total,
                posterior_target=self.posterior_target,
                n_masks=lpaa_n_masks,
                patch_size=lpaa_patch_size,
                keep_frac=lpaa_keep_frac,
                clamp_min=self.clamp_min,
                clamp_max=self.clamp_max,
            )
        else:
            l_vae = l_vae_full

        if noise_mode=="mask":
            l_noise = noise_loss(img_orig, img_final, 1 - mask, noise_on_mask=True)
        elif noise_mode =="all":
            l_noise = noise_loss(img_orig, img_final, 1 - mask, noise_on_mask=False)
        else:
            raise ValueError("noise mode does not match with all or mask")

        loss    = lambda_vae * l_vae + lambda_noise * l_noise 

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step == 0 or (step + 1) % log_every == 0:
            delta_abs = (img_final - img_orig).abs()
            if use_lpaa:
                lpaa_tag = f"l_vae_lpaa={l_vae.item():.6f}  l_vae_full={l_vae_full.item():.6f}  "
            else:
                lpaa_tag = f"l_vae={l_vae.item():.6f}  "
            print(
                f"  [UNet step {step+1:3d}/{n_steps}] "
                f"loss={loss.item():.6f}  "
                f"{lpaa_tag}"
                f"l_noise={l_noise.item():.6f}  "
                f"cosine_dist={cosine_dist:.4f}  "
                f"delta_mean={delta_abs.mean().item():.6f}  "
                f"delta_max={delta_abs.max().item():.6f}"
            ) 

     with torch.no_grad():
        noise     = local_unet(img_adv)
        if noise_mode=="mask":
            noise    = noise * (1 - mask)
        noise     = torch.clamp(noise, -eps, eps)
        img_final = img_adv + noise
        total_delta = torch.clamp(img_final - img_orig, -eps, eps)
        img_final   = torch.clamp(img_orig + total_delta, -1.0, 1.0)

     print(
        f"  [UNet refinement done] "
        f"delta_mean={(img_final - img_orig).abs().mean().item():.6f}  "
        f"delta_max={(img_final - img_orig).abs().max().item():.6f}"
     )

     return img_final.detach(), l_vae, l_noise
   

    def immunize_img_targeted(
    self,
    img: torch.Tensor,
    img_mask: torch.Tensor,
    noise_mode: str= "mask",
    is_2_stage: bool = True,
    pgd : bool = True,
    eps: float = 32/255 ,
    lr: float = 1e-4,
    n_steps: int = 100,
    lambda_vae: float = 1.0,
    lambda_noise: float = 50.0,
    # --- LPAA patch-based ---
    use_lpaa = True,
    lpaa_n_masks: int = 5,
    lpaa_patch_size: int = 256,
    lpaa_keep_frac: float = 1,
    run_diagnostic: bool = True,
    ) -> torch.Tensor:

     if is_2_stage:
        img_adv = self.immunize_img(img, img_mask,noise=noise_mode)

        img_final, l_vae, l_noise = self.targeted_unet_refinement(
            img=img_adv,
            img_orig=img,
            img_mask=img_mask,
            noise_mode=noise_mode,
            eps=eps,
            lr=lr,
            n_steps=n_steps,
            lambda_vae=lambda_vae,
            lambda_noise=lambda_noise,
            use_lpaa=use_lpaa,
            lpaa_n_masks=lpaa_n_masks,
            lpaa_patch_size=lpaa_patch_size,
            lpaa_keep_frac=lpaa_keep_frac,
        )

        if run_diagnostic:
            from diagnose_local_overfit import run_full_diagnostic

            results = run_full_diagnostic(
                vae=self.vae,
                img_orig=img,
                img_final=img_final,
                target_mean=self.posterior_target.mean,
            )

        return img_final, l_vae.item(), l_noise.item()

     img_final = self.immunize_img(img, img_mask, noise=noise_mode)
     l_vae, l_noise = 0,0

     return img_final, l_vae, l_noise

    

class Attack:
    def __init__(self, model_link: str = "runwayml/stable-diffusion-inpainting", scheduler: str = "DDIM", local_files_only: bool = True):
        try:
            pipe_inpaint = StableDiffusionInpaintPipeline.from_pretrained(
                model_link,
                torch_dtype=torch.float,
                local_files_only=local_files_only,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Could not load the model '{model_link}' from local cache. "
                "If you have network access, set local_files_only=False or cache the model locally under HF_HOME."
            ) from exc

        if scheduler == "DDIM":
            pipe_inpaint.scheduler = DDIMScheduler.from_config(
                pipe_inpaint.scheduler.config
            )

        self.model = pipe_inpaint.to("cuda:0")
        self.model_link = model_link
        self.generator = torch.Generator(device="cuda:0")

    def edit_image(self, prompt, img, img_mask, num_inf=30, SEED=5):
        """Edit image using SD Inpainting pipeline."""
        self.generator.manual_seed(SEED)
        edited_image = self.model(
            prompt=prompt,
            image=img,
            mask_image=img_mask,
            eta=1,
            num_inference_steps=num_inf,
            guidance_scale=7.5,
            strength=1.0,
            generator=self.generator,
        ).images
        return edited_image


class AttackInstructPix2Pix:
    """Wrapper per InstructPix2Pix compatibile con DiffVax."""

    def __init__(self, model_id="timbrooks/instruct-pix2pix"):
        self.model = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float,
            safety_checker=None,
        ).to("cuda:0")

    def edit_image(self, prompt, image, mask=None, num_inference_steps=30,
                   image_guidance_scale=1.5, guidance_scale=7.5):
        """
        Edita l'immagine con un'istruzione testuale.
        mask è ignorata (InstructPix2Pix non la usa) ma mantenuta
        per compatibilità con l'interfaccia DiffVax.
        """
        result = self.model(
            prompt=prompt,
            image=image,
            num_inference_steps=num_inference_steps,
            image_guidance_scale=image_guidance_scale,  # fedeltà all'originale
            guidance_scale=guidance_scale,              # fedeltà al testo
        )
        return result.images


class AttackSD:
    """Stable Diffusion image editing wrapper using img2img."""

    def __init__(self, model_link: str = "stable-diffusion-v1-5/stable-diffusion-v1-5", scheduler: str = "DDIM"):
        self.pipe = AutoPipelineForImage2Image.from_pretrained(
            model_link,
            torch_dtype=torch.float,
            use_safetensors=True,
            local_files_only=False,
        )

        self.model = self.pipe.to("cuda:0")
        self.generator = torch.Generator(device="cuda:0")

    def edit_image(
        self,
        prompt: Union[str, List[str]],
        image: Union[torch.FloatTensor, Image.Image],
        mask: None = None,
        num_inference_steps: int = 30,
        strength: float = 0.5,
        seed: int = 5,
    ):
        """Edita l'immagine con Stable Diffusion Img2Img."""

        # Converte il tensore in PIL Image se necessario
        if isinstance(image, torch.FloatTensor):
            image = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
            image = Image.fromarray((image * 255).astype("uint8"))

        # Ridimensiona a multipli di 8
        w, h = image.size
        w, h = (x - x % 8 for x in (w, h))
        image = image.resize((w, h), Image.LANCZOS)

        self.generator.manual_seed(seed)

        result = self.model(
            prompt=prompt,
            image=image,
            strength=strength,
            num_inference_steps=num_inference_steps,
            guidance_scale=7.5,
            generator=self.generator,
        )

        return result.images


from diffusers import AutoPipelineForImage2Image

class AttackSDXL:
    """Stable Diffusion XL img2img editing wrapper."""

    def __init__(self, model_link: str = "stabilityai/stable-diffusion-xl-base-1.0"):
        self.pipe = AutoPipelineForImage2Image.from_pretrained(
            model_link,
            torch_dtype=torch.float,
            use_safetensors=True,
            local_files_only=True,
            variant="fp16",
        ).to("cuda:0")
        self.generator = torch.Generator(device="cuda:0")
        self.model = self.pipe

    def edit_image(self, prompt, image, mask=None, num_inference_steps=50,
                   strength=0.8, seed=2043):

        if isinstance(image, torch.FloatTensor):
            image = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
            image = Image.fromarray((image * 255).astype("uint8"))

        w, h = image.size
        original_size = (w, h)
        w, h = (x - x % 64 for x in (w, h))
        image = image.resize((w, h), Image.LANCZOS)

        self.generator.manual_seed(seed)

        result = self.pipe(
            prompt=prompt,
            image=image,
            strength=strength,
            num_inference_steps=num_inference_steps,
            guidance_scale=12.0,
            generator=self.generator,
        )

        output = result.images[0]
        output = output.resize(original_size, Image.LANCZOS)  # riporta a dimensione originale
        return [output]