"""InstructPix2Pix model handler for DiffuseMix.

Wraps the StableDiffusionInstructPix2PixPipeline from diffusers.
"""

import torch
from PIL import Image
from diffusers import (
    StableDiffusionInstructPix2PixPipeline,
    EulerAncestralDiscreteScheduler,
)


class ModelHandler:
    """Manages the InstructPix2Pix diffusion pipeline.

    Args:
        model_id: HuggingFace model identifier.
                  Default: "timbrooks/instruct-pix2pix".
        device: Torch device ('cuda' or 'cpu').
    """

    def __init__(self, model_id="timbrooks/instruct-pix2pix", device='cuda'):
        print(f"[DiffuseMix] Loading InstructPix2Pix model: {model_id}")
        self.device = device
        self.pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            use_safetensors=True,
            safety_checker=None,
        ).to(device)
        self.pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
            self.pipeline.scheduler.config
        )
        print("[DiffuseMix] Model loaded successfully.")

    def generate_images(self, prompt, img_path, num_images=1,
                        guidance_scale=4, size=(256, 256)):
        """Generate style-transferred images from a file path.

        Args:
            prompt: Style prompt (e.g., "sunset").
            img_path: Path to input image.
            num_images: Number of images to generate.
            guidance_scale: Classifier-free guidance scale.
            size: Resolution for the input image.

        Returns:
            List of PIL.Image results.
        """
        image = Image.open(img_path).convert('RGB').resize(size)
        return self.pipeline(
            prompt,
            image=image,
            num_images_per_prompt=num_images,
            guidance_scale=guidance_scale,
        ).images

    def generate_images_from_pil(self, prompt, pil_image, num_images=1,
                                  guidance_scale=4, size=(256, 256)):
        """Generate style-transferred images from a PIL image.

        Args:
            prompt: Style prompt.
            pil_image: PIL.Image input.
            num_images: Number of images to generate.
            guidance_scale: Classifier-free guidance scale.
            size: Resolution for the input image.

        Returns:
            List of PIL.Image results.
        """
        image = pil_image.convert('RGB').resize(size)
        return self.pipeline(
            prompt,
            image=image,
            num_images_per_prompt=num_images,
            guidance_scale=guidance_scale,
        ).images
