from torch import nn
from PIL import Image
import torch
from diffusers import (
    StableDiffusionInpaintPipeline,
    AutoPipelineForImage2Image,
    DDIMScheduler
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
        device:         str   = "cuda:0",
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
    def __init__(self, model_link: str = "runwayml/stable-diffusion-inpainting", scheduler: str = "DDIM", local_files_only: bool = True):
        try:
            pipe_inpaint = StableDiffusionInpaintPipeline.from_pretrained(
                model_link,
                torch_dtype=torch.float16,
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
        self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            safety_checker=None,
        ).to("cuda:0")

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


class AttackSD:
    """Stable Diffusion image editing wrapper using img2img."""

    def __init__(self, model_link: str = "stable-diffusion-v1-5/stable-diffusion-v1-5", scheduler: str = "DDIM"):
        self.pipe = AutoPipelineForImage2Image.from_pretrained(
            model_link,
            torch_dtype=torch.float16,
            use_safetensors=True,
            local_files_only=False,
        )

        self.pipe = self.pipe.to("cuda:0")
        self.generator = torch.Generator(device="cuda:0")

    def edit_image(
        self,
        prompt: Union[str, List[str]],
        image: Union[torch.FloatTensor, Image.Image],
        mask: None = None,
        num_inference_steps: int = 30,
        strength: float = 0.8,
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

        result = self.pipe(
            prompt=prompt,
            image=image,
            strength=strength,
            num_inference_steps=num_inference_steps,
            guidance_scale=7.5,
            generator=self.generator,
        )

        return result.images
   