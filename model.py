from torch import nn
from PIL import Image
import torch
from diffusers import (
    StableDiffusionInpaintPipeline,
    DDIMScheduler,
)
from typing import Union, List, Optional, Callable
from diffusers import StableDiffusionInstructPix2PixPipeline

class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

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


class DiffVaxImmunization:
    def __init__(
        self,
        device:         str   = "cuda:1",
        clamp_min:      float = -1.0,
        clamp_max:      float =  1.0,
        load_existing:  bool  = False,
        load_path:      str   = None,
    ):
        self.device    = device
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

        self.model = NestedUNet(num_classes=3).to(device)

        if load_existing:
            if load_path is None:
                raise ValueError("load_path must be specified when load_existing=True")
            self.model.load_state_dict(torch.load(load_path, weights_only=True, map_location=device))
            print(f"Checkpoint loaded from {load_path}")

        self.model.eval()

    def immunize_img(self, img: torch.Tensor, img_mask: torch.Tensor) -> torch.Tensor:
        """
        Applica la perturbazione di immunizzazione all'immagine.

        Args:
            img      : immagine originale  (B, 3, H, W)
            img_mask : maschera binaria    (B, 1, H, W)
        Returns:
            img_adv  : immagine immunizzata (B, 3, H, W)
        """
        img_f  = img.float().to(self.device)
        mask_f = img_mask.float().to(self.device)

        with torch.no_grad():
            unet_out = self.model(img_f)
            unet_out = unet_out * (1 - mask_f)

        img_adv = torch.clamp(img_f + unet_out, self.clamp_min, self.clamp_max)
        return img_adv


class Attack:
    def __init__(self, model_link: str, scheduler: str = "DDIM"):
        pipe_inpaint = StableDiffusionInpaintPipeline.from_pretrained(
            model_link,
            torch_dtype=torch.float16,
            local_files_only=True,
        )
        if scheduler == "DDIM":
            pipe_inpaint.scheduler = DDIMScheduler.from_config(
                pipe_inpaint.scheduler.config
            )

        self.model = pipe_inpaint.to("cuda:1")
        self.model_link = model_link
        self.generator = torch.Generator(device="cuda:1")

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

    def attack(
        self,
        prompt: Union[str, List[str]],
        masked_image: Union[torch.FloatTensor, Image.Image],
        mask: Union[torch.FloatTensor, Image.Image],
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        eta: float = 0.0,
        batch_size: int = 1,
    ):
        """Differentiable forward pass of the inpainting stable diffusion model."""
        diffusion_model = self.model

        text_embeddings = self.tokenize_prompt(
            diffusion_model, prompt, batch_size=batch_size
        )

        num_channels_latents = diffusion_model.vae.config.latent_channels

        latents_shape = (
            batch_size,
            num_channels_latents,
            height // 8,
            width // 8,
        )
        latents = torch.randn(
            latents_shape,
            device=diffusion_model.device,
            dtype=text_embeddings.dtype,
        )

        mask = torch.nn.functional.interpolate(mask, size=(height // 8, width // 8))
        mask = torch.cat([mask] * 2)

        masked_image_latents = diffusion_model.vae.encode(
            masked_image
        ).latent_dist.sample()
        masked_image_latents = 0.18215 * masked_image_latents
        masked_image_latents = torch.cat([masked_image_latents] * 2)

        latents = latents * diffusion_model.scheduler.init_noise_sigma

        diffusion_model.scheduler.set_timesteps(num_inference_steps)
        timesteps_tensor = diffusion_model.scheduler.timesteps.to(
            diffusion_model.device
        )

        for i, t in enumerate(timesteps_tensor):
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = torch.cat(
                [latent_model_input, mask, masked_image_latents], dim=1
            )
            noise_pred = diffusion_model.unet(
                latent_model_input, t, encoder_hidden_states=text_embeddings
            ).sample
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

            latents = diffusion_model.scheduler.step(noise_pred).prev_sample

        latents = 1 / 0.18215 * latents
        image = diffusion_model.vae.decode(latents).sample
        return image

    def tokenize_prompt(
        self, diffusion_model, prompt, batch_size=1, tokenize_negative=False
    ):
        """Tokenize prompts. Uses 'gray background' as unconditional embedding if tokenize_negative is True."""
        text_inputs = diffusion_model.tokenizer(
            prompt,
            padding="max_length",
            max_length=diffusion_model.tokenizer.model_max_length,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        text_embeddings = diffusion_model.text_encoder(
            text_input_ids.to(diffusion_model.device)
        )[0]

        uncond_tokens = [""] * batch_size
        if tokenize_negative:
            uncond_tokens = ["gray background"]
        max_length = text_input_ids.shape[-1]
        uncond_input = diffusion_model.tokenizer(
            uncond_tokens,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        uncond_embeddings = diffusion_model.text_encoder(
            uncond_input.input_ids.to(diffusion_model.device)
        )[0]
        seq_len = uncond_embeddings.shape[1]
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        text_embeddings = text_embeddings.detach()
        return text_embeddings

class AttackInstructPix2Pix:
    """Wrapper per InstructPix2Pix compatibile con DiffVax."""

    def __init__(self, model_id="timbrooks/instruct-pix2pix"):
        self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            safety_checker=None,
        ).to("cuda:1")

    def edit_image(self, prompt, image, mask=None, num_inference_steps=30,
                   image_guidance_scale=1.5, guidance_scale=7.5):
        """
        Edita l'immagine con un'istruzione testuale.
        mask è ignorata (InstructPix2Pix non la usa) ma mantenuta
        per compatibilità con l'interfaccia DiffVax.
        """
        result = self.pipe(
            prompt=prompt,
            image=image,
            num_inference_steps=num_inference_steps,
            image_guidance_scale=image_guidance_scale,  # fedeltà all'originale
            guidance_scale=guidance_scale,              # fedeltà al testo
        )
        return result.images
