"""SDEdit-based image generator for Head-to-Tail transfer.

Core generation mechanism:
  1. Take a head-class image as input
  2. Add noise to it (controlled by `strength` parameter)
  3. Denoise using a tail-class prompt via Stable Diffusion img2img
  4. The result preserves low-level domain features from the head image
     while adopting the semantic content of the tail class

Key parameters:
  - strength (0.0-1.0): Controls how much of the original image to preserve
    - Low (0.3-0.5): Preserves structure → good domain consistency, risk of
      head-class semantics leaking through
    - Medium (0.5-0.7): Balance point → recommended starting range
    - High (0.8-1.0): Near-pure generation → less domain advantage
  - guidance_scale: Controls how strongly the text prompt guides generation
  - image_guidance_scale: (if using InstructPix2Pix) Controls faithfulness
    to the input image
"""

import os
import torch
import numpy as np
from PIL import Image


class Head2TailGenerator:
    """SDEdit-based generator that transforms head-class images into
    tail-class images.

    Supports two pipelines:
      1. StableDiffusionImg2ImgPipeline (SDEdit): Standard img2img with
         noise injection + tail-class prompt denoising.
      2. StableDiffusionInstructPix2PixPipeline: Instruction-based editing.

    Args:
        model_id: HuggingFace model ID.
        pipeline_type: 'img2img' (recommended) or 'pix2pix'.
        device: Torch device.
        lora_weights: Path to LoRA weights (optional).
    """

    def __init__(self, model_id="runwayml/stable-diffusion-v1-5",
                 pipeline_type='img2img', device='cuda',
                 lora_weights=None):
        self.device = device
        self.pipeline_type = pipeline_type

        print(f"[Head2Tail] Loading {pipeline_type} pipeline: {model_id}")

        if pipeline_type == 'img2img':
            from diffusers import (
                StableDiffusionImg2ImgPipeline,
                EulerAncestralDiscreteScheduler,
            )
            self.pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                safety_checker=None,
            ).to(device)
            self.pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
                self.pipeline.scheduler.config
            )

        elif pipeline_type == 'pix2pix':
            from diffusers import (
                StableDiffusionInstructPix2PixPipeline,
                EulerAncestralDiscreteScheduler,
            )
            self.pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                safety_checker=None,
            ).to(device)
            self.pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
                self.pipeline.scheduler.config
            )

        else:
            raise ValueError(f"Unknown pipeline type: {pipeline_type}")

        # Load LoRA weights if provided
        if lora_weights and os.path.exists(lora_weights):
            self._load_lora_weights(lora_weights)

        # Enable memory optimizations
        try:
            self.pipeline.enable_xformers_memory_efficient_attention()
        except Exception:
            pass  # xformers not available

        print("[Head2Tail] Generator ready.")

    def _load_lora_weights(self, lora_weights):
        """Load LoRA weights with robust fallback.

        Preferred path is diffusers' native loader. If LoRA was saved via
        PEFT UNet (keys like base_model.model....), fallback to PEFT merge.
        """
        print(f"[Head2Tail] Loading LoRA weights: {lora_weights}")

        # 1) Try native diffusers loader first
        try:
            self.pipeline.load_lora_weights(lora_weights)
            print("[Head2Tail] LoRA weights loaded (diffusers native).")
            return
        except Exception as e:
            print(f"[Head2Tail][WARN] Native LoRA load failed: {e}")
            print("[Head2Tail] Trying PEFT fallback loader...")

        # 2) Fallback: PEFT adapter load + merge into UNet
        try:
            from peft import PeftModel

            peft_unet = PeftModel.from_pretrained(
                self.pipeline.unet,
                lora_weights,
                is_trainable=False,
            )
            self.pipeline.unet = peft_unet.merge_and_unload()
            self.pipeline.unet.to(self.device, dtype=torch.float16)
            print("[Head2Tail] LoRA weights loaded via PEFT fallback (merged).")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load LoRA weights from '{lora_weights}'. "
                f"Native and PEFT fallback both failed. Last error: {e}"
            )

    def generate(self, head_image, prompt, strength=0.6,
                 guidance_scale=7.5, negative_prompt=None,
                 num_images=1, size=(512, 512),
                 num_inference_steps=50):
        """Generate tail-class images from a head-class image via SDEdit.

        Args:
            head_image: PIL.Image (the source head-class image).
            prompt: Text prompt describing the target tail class
                    (e.g., "a photo of a leopard").
            strength: Noise strength (0.0 = no change, 1.0 = complete regen).
                      Controls how much of the original image to preserve.
            guidance_scale: Classifier-free guidance scale.
            negative_prompt: Negative prompt to avoid unwanted features.
            num_images: Number of images to generate.
            size: Generation resolution (width, height).
            num_inference_steps: Number of denoising steps.

        Returns:
            List[PIL.Image]: Generated images.
        """
        if negative_prompt is None:
            negative_prompt = (
                "blurry, low quality, artifacts, distorted, "
                "watermark, text, deformed"
            )

        # Resize input
        head_image = head_image.convert('RGB').resize(size)

        if self.pipeline_type == 'img2img':
            result = self.pipeline(
                prompt=prompt,
                image=head_image,
                strength=strength,
                guidance_scale=guidance_scale,
                negative_prompt=negative_prompt,
                num_images_per_prompt=num_images,
                num_inference_steps=num_inference_steps,
            )

        elif self.pipeline_type == 'pix2pix':
            # For InstructPix2Pix, use instruction-style prompt
            result = self.pipeline(
                prompt=prompt,
                image=head_image,
                guidance_scale=guidance_scale,
                image_guidance_scale=1.5,
                num_images_per_prompt=num_images,
                num_inference_steps=num_inference_steps,
            )

        return result.images

    def generate_batch(self, head_images, prompts, strength=0.6,
                       guidance_scale=7.5, negative_prompt=None,
                       size=(512, 512), num_inference_steps=50):
        """Generate tail-class images from multiple head-class images.

        Each (head_image, prompt) pair produces one generated image.

        Args:
            head_images: List[PIL.Image] source images.
            prompts: List[str] or str. If str, same prompt for all.
            strength: float or List[float].
            guidance_scale: float.
            negative_prompt: str.
            size: (width, height).
            num_inference_steps: int.

        Returns:
            List[PIL.Image]: Generated images (one per input).
        """
        if isinstance(prompts, str):
            prompts = [prompts] * len(head_images)
        if isinstance(strength, float):
            strength_list = [strength] * len(head_images)
        else:
            strength_list = strength

        results = []
        for img, prompt, s in zip(head_images, prompts, strength_list):
            gen = self.generate(
                img, prompt, strength=s,
                guidance_scale=guidance_scale,
                negative_prompt=negative_prompt,
                num_images=1, size=size,
                num_inference_steps=num_inference_steps
            )
            results.extend(gen)

        return results

    @staticmethod
    def is_valid_image(image, black_threshold=0.9):
        """Check if a generated image is valid (not all black/white)."""
        arr = np.array(image.convert('L'))
        total = arr.size
        # Check for mostly black
        if (arr < 10).sum() > black_threshold * total:
            return False
        # Check for mostly white
        if (arr > 245).sum() > black_threshold * total:
            return False
        return True

    def generate_with_retry(self, head_image, prompt, strength=0.6,
                            guidance_scale=7.5, negative_prompt=None,
                            size=(512, 512), max_retries=3,
                            num_inference_steps=50):
        """Generate with automatic retry on invalid outputs.

        Args:
            head_image: PIL.Image.
            prompt: str.
            strength: float.
            guidance_scale: float.
            negative_prompt: str.
            size: (width, height).
            max_retries: Maximum number of retries.
            num_inference_steps: int.

        Returns:
            PIL.Image or None if all retries failed.
        """
        for attempt in range(max_retries):
            images = self.generate(
                head_image, prompt, strength=strength,
                guidance_scale=guidance_scale,
                negative_prompt=negative_prompt,
                num_images=1, size=size,
                num_inference_steps=num_inference_steps,
            )
            if images and self.is_valid_image(images[0]):
                return images[0]
            # Vary strength slightly on retry
            strength = min(0.95, strength + 0.05)

        return None
