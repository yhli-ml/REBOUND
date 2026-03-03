"""Augmentation module for long-tail learning.

Contains two augmentation pipelines:

1. DiffuseMix (Islam et al., CVPR 2024):
   Style transfer via InstructPix2Pix + concatenation + fractal blending.

2. Head-to-Tail Transfer (ours):
   Feature-guided cross-class image transfer via SDEdit.
   Uses head-class images as domain-consistent seeds for tail-class generation.

Both follow a two-stage pipeline:
  Stage 1 (Offline): Generate augmented images for tail classes.
  Stage 2 (Online): Wrap original + augmented images for training.
"""

from .diffusemix_utils import DiffuseMixTransform, load_fractal_images
from .diffusemix_handler import ModelHandler
from .head2tail_prompts import get_class_names, get_all_prompts, get_prompts_for_class
from .head2tail_selector import HeadClassSelector
from .head2tail_generator import Head2TailGenerator

__all__ = [
    # DiffuseMix
    'DiffuseMixTransform',
    'load_fractal_images',
    'ModelHandler',
    # Head-to-Tail
    'HeadClassSelector',
    'Head2TailGenerator',
    'get_class_names',
    'get_all_prompts',
    'get_prompts_for_class',
]
