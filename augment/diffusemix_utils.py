"""Core DiffuseMix image manipulation utilities.

Implements the three key operations from the DiffuseMix paper:
  1. Style-transfer via InstructPix2Pix (handled by ModelHandler)
  2. Concatenation: blend original + generated image (half-half)
  3. Fractal blending: overlay a fractal image at low opacity
"""

import os
import random
import numpy as np
from PIL import Image


def load_fractal_images(fractal_dir, size=(256, 256)):
    """Load all fractal images from a directory and resize.

    Args:
        fractal_dir: Path to directory containing fractal .jpg/.png images.
        size: Target (width, height) to resize each image.

    Returns:
        List of PIL.Image in RGB mode.
    """
    exts = ('.png', '.jpg', '.jpeg')
    paths = sorted([
        os.path.join(fractal_dir, f)
        for f in os.listdir(fractal_dir) if f.lower().endswith(exts)
    ])
    if not paths:
        raise FileNotFoundError(f"No fractal images found in {fractal_dir}")
    images = [Image.open(p).convert('RGB').resize(size) for p in paths]
    print(f"[DiffuseMix] Loaded {len(images)} fractal images from {fractal_dir}")
    return images


def combine_images(original_img, augmented_img, blend_width=20):
    """Concatenate original and augmented images with smooth blending.

    Randomly choose horizontal or vertical split, then smoothly blend
    across `blend_width` pixels at the boundary.

    Args:
        original_img: PIL.Image (RGB).
        augmented_img: PIL.Image (RGB), same size as original.
        blend_width: Width of the smooth transition zone.

    Returns:
        Blended PIL.Image.
    """
    width, height = original_img.size
    choice = random.choice(['horizontal', 'vertical'])

    if choice == 'vertical':
        mask = np.linspace(0, 1, blend_width).reshape(-1, 1)
        mask = np.tile(mask, (1, width))
        mask = np.vstack([
            np.zeros((height // 2 - blend_width // 2, width)),
            mask,
            np.ones((height // 2 - blend_width // 2 + blend_width % 2, width)),
        ])
        mask = np.tile(mask[:, :, np.newaxis], (1, 1, 3))
    else:
        mask = np.linspace(0, 1, blend_width).reshape(1, -1)
        mask = np.tile(mask, (height, 1))
        mask = np.hstack([
            np.zeros((height, width // 2 - blend_width // 2)),
            mask,
            np.ones((height, width // 2 - blend_width // 2 + blend_width % 2)),
        ])
        mask = np.tile(mask[:, :, np.newaxis], (1, 1, 3))

    orig_arr = np.array(original_img, dtype=np.float32) / 255.0
    aug_arr = np.array(augmented_img, dtype=np.float32) / 255.0
    blended = (1 - mask) * orig_arr + mask * aug_arr
    blended = np.clip(blended * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(blended)


def blend_with_fractal(image, fractal_img, alpha=0.20):
    """Blend image with a fractal overlay.

    Args:
        image: PIL.Image (RGB).
        fractal_img: PIL.Image (RGB), will be resized to match.
        alpha: Blending opacity for the fractal (0 = no fractal).

    Returns:
        Blended PIL.Image.
    """
    fractal_resized = fractal_img.resize(image.size)
    base = np.array(image, dtype=np.float32)
    overlay = np.array(fractal_resized, dtype=np.float32)
    blended = (1 - alpha) * base + alpha * overlay
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    return Image.fromarray(blended)


def is_black_image(image, threshold=0.9):
    """Check if an image is predominantly black (failed generation)."""
    hist = image.convert("L").histogram()
    total = image.size[0] * image.size[1]
    return hist[-1] > threshold * total


class DiffuseMixTransform:
    """Apply the full DiffuseMix pipeline to a single image.

    Steps:
      1. Generate style-transferred version via InstructPix2Pix.
      2. Concatenate original + generated (half-half blend).
      3. Blend with random fractal image.

    This class is used by the offline generation script.
    """

    def __init__(self, model_handler, fractal_imgs, prompts,
                 guidance_scale=4, gen_size=(256, 256), fractal_alpha=0.20):
        """
        Args:
            model_handler: ModelHandler instance with `generate_images()`.
            fractal_imgs: List of PIL.Image fractal overlays.
            prompts: List of style prompts (e.g., ["sunset", "Autumn"]).
            guidance_scale: InstructPix2Pix guidance scale.
            gen_size: Resolution for generation (width, height).
            fractal_alpha: Fractal blending opacity.
        """
        self.model_handler = model_handler
        self.fractal_imgs = fractal_imgs
        self.prompts = prompts
        self.guidance_scale = guidance_scale
        self.gen_size = gen_size
        self.fractal_alpha = fractal_alpha

    def __call__(self, img_path_or_pil, return_all=False):
        """Generate augmented images for one input.

        Args:
            img_path_or_pil: Path to image file or PIL.Image.
            return_all: If True, return one augmented image per prompt.
                        If False, return one randomly selected.

        Returns:
            List of PIL.Image (if return_all) or single PIL.Image.
        """
        if isinstance(img_path_or_pil, str):
            original = Image.open(img_path_or_pil).convert('RGB')
        else:
            original = img_path_or_pil.convert('RGB')

        original_resized = original.resize(self.gen_size)
        results = []

        for prompt in self.prompts:
            # Step 1: Style transfer via InstructPix2Pix
            if isinstance(img_path_or_pil, str):
                generated_list = self.model_handler.generate_images(
                    prompt, img_path_or_pil, num_images=1,
                    guidance_scale=self.guidance_scale)
            else:
                generated_list = self.model_handler.generate_images_from_pil(
                    prompt, original_resized, num_images=1,
                    guidance_scale=self.guidance_scale)

            for gen_img in generated_list:
                gen_img = gen_img.resize(self.gen_size)
                if is_black_image(gen_img):
                    continue

                # Step 2: Concatenate original + generated
                combined = combine_images(original_resized, gen_img)

                # Step 3: Blend with random fractal
                fractal = random.choice(self.fractal_imgs)
                blended = blend_with_fractal(combined, fractal, self.fractal_alpha)
                results.append(blended)

        if not results:
            # Fallback: return original if all generations failed
            return [original_resized] if return_all else original_resized

        if return_all:
            return results
        return random.choice(results)
