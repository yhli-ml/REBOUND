"""Class-aware prompt templates for Head-to-Tail transfer.

Instead of generic style prompts ("sunset", "Autumn"), we use prompts that
describe the TARGET class in different scenarios, preserving semantic identity
while encouraging visual diversity.

Prompt hierarchy:
  Level 1: "a photo of a {class_name}"  (baseline)
  Level 2: "{template} {class_name} {context}"  (diverse descriptions)
  Level 3: Custom per-class prompts (if provided via JSON)

Supports: CIFAR-10, CIFAR-100, ImageNet-LT
"""

import json
import os
import random

# ============================================================
# CIFAR-10 class names (index → name)
# ============================================================
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck',
]

# ============================================================
# CIFAR-100 fine-label class names (index → name)
# ============================================================
CIFAR100_CLASSES = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver',
    'bed', 'bee', 'beetle', 'bicycle', 'bottle',
    'bowl', 'boy', 'bridge', 'bus', 'butterfly',
    'camel', 'can', 'castle', 'caterpillar', 'cattle',
    'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach',
    'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox',
    'girl', 'hamster', 'house', 'kangaroo', 'keyboard',
    'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard',
    'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
    'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid',
    'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree',
    'plain', 'plate', 'poppy', 'porcupine', 'possum',
    'rabbit', 'raccoon', 'ray', 'road', 'rocket',
    'rose', 'sea', 'seal', 'shark', 'shrew',
    'skunk', 'skyscraper', 'snail', 'snake', 'spider',
    'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
    'tank', 'telephone', 'television', 'tiger', 'tractor',
    'train', 'trout', 'tulip', 'turtle', 'wardrobe',
    'whale', 'willow_tree', 'wolf', 'woman', 'worm',
]

# ============================================================
# Diverse prompt templates
# Each template produces a semantically consistent but visually
# diverse description of the target class.
# ============================================================
PROMPT_TEMPLATES = [
    "a photo of a {cls}",
    "a clear photo of a {cls}",
    "a photo of a {cls} in natural lighting",
    "a photo of a {cls} from a different angle",
    "a photo of a small {cls}",
    "a photo of a large {cls}",
    "a close-up photo of a {cls}",
    "a photo of a {cls} with a simple background",
    "a photo of a {cls} outdoors",
    "a photo of a {cls} indoors",
]

# Additional templates for living things (animals, people)
LIVING_TEMPLATES = [
    "a photo of a {cls} in its natural habitat",
    "a photo of a {cls} resting",
    "a photo of a {cls} in motion",
    "a photo of a young {cls}",
]

# Additional templates for objects / vehicles
OBJECT_TEMPLATES = [
    "a photo of a {cls} on a road",
    "a photo of a {cls} in a parking lot",
    "a photo of a new {cls}",
    "a photo of a {cls} from the side",
]

# Set of "living thing" classes (animals, people, insects)
_LIVING_CLASSES = {
    # CIFAR-10
    'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
    # CIFAR-100 animals & people
    'aquarium_fish', 'baby', 'bear', 'beaver', 'bee', 'beetle', 'boy',
    'butterfly', 'camel', 'caterpillar', 'cattle', 'chimpanzee', 'cockroach',
    'crab', 'crocodile', 'dinosaur', 'dolphin', 'elephant', 'flatfish',
    'fox', 'girl', 'hamster', 'kangaroo', 'leopard', 'lion', 'lizard',
    'lobster', 'man', 'mouse', 'otter', 'porcupine', 'possum', 'rabbit',
    'raccoon', 'ray', 'seal', 'shark', 'shrew', 'skunk', 'snail', 'snake',
    'spider', 'squirrel', 'tiger', 'trout', 'turtle', 'whale', 'wolf',
    'woman', 'worm',
}

_OBJECT_CLASSES = {
    'airplane', 'automobile', 'ship', 'truck',
    'bicycle', 'bottle', 'bus', 'can', 'motorcycle', 'pickup_truck',
    'streetcar', 'tank', 'tractor', 'train', 'rocket',
}


def get_class_names(dataset_name):
    """Return list of class names for a dataset.

    Args:
        dataset_name: 'cifar10_lt', 'cifar100_lt', or 'imagenet_lt'

    Returns:
        List[str] of class names, indexed by class label.
    """
    if dataset_name == 'cifar10_lt':
        return list(CIFAR10_CLASSES)
    elif dataset_name == 'cifar100_lt':
        return list(CIFAR100_CLASSES)
    elif dataset_name == 'imagenet_lt':
        # Try loading from a JSON file; fall back to generic names
        imagenet_names_path = os.path.join(
            os.path.dirname(__file__), '..', 'configs', 'imagenet_class_names.json'
        )
        if os.path.exists(imagenet_names_path):
            with open(imagenet_names_path) as f:
                return json.load(f)
        else:
            return [f'class_{i}' for i in range(1000)]
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")


def get_prompts_for_class(class_name, n_prompts=5, seed=None):
    """Generate diverse prompts for a specific class.

    Args:
        class_name: The target class name (e.g., 'leopard').
        n_prompts: Number of prompts to return.
        seed: Random seed for reproducibility (None = random).

    Returns:
        List[str] of prompts.
    """
    rng = random.Random(seed)

    # Build template pool based on class type
    templates = list(PROMPT_TEMPLATES)
    display_name = class_name.replace('_', ' ')

    if class_name in _LIVING_CLASSES:
        templates.extend(LIVING_TEMPLATES)
    elif class_name in _OBJECT_CLASSES:
        templates.extend(OBJECT_TEMPLATES)

    # Sample without replacement if possible
    n = min(n_prompts, len(templates))
    selected = rng.sample(templates, n)

    return [t.format(cls=display_name) for t in selected]


def get_all_prompts(dataset_name, n_prompts_per_class=5, seed=42):
    """Generate prompts for ALL classes in a dataset.

    Args:
        dataset_name: 'cifar10_lt', 'cifar100_lt', 'imagenet_lt'.
        n_prompts_per_class: Number of prompts per class.
        seed: Random seed.

    Returns:
        Dict[int, List[str]]: {class_idx: [prompt1, prompt2, ...]}
    """
    class_names = get_class_names(dataset_name)
    all_prompts = {}
    for idx, name in enumerate(class_names):
        all_prompts[idx] = get_prompts_for_class(
            name, n_prompts=n_prompts_per_class, seed=seed + idx
        )
    return all_prompts


def load_custom_prompts(json_path):
    """Load custom per-class prompts from a JSON file.

    Expected format:
        {
            "0": ["prompt1 for class 0", "prompt2 for class 0"],
            "1": ["prompt1 for class 1", ...],
            ...
        }

    Args:
        json_path: Path to JSON file.

    Returns:
        Dict[int, List[str]]
    """
    with open(json_path) as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}

# Use GPT-generated prompts
def generate_gpt_prompts(dataset_name, n_prompts_per_class=5, seed=42):
    """Generate prompts using GPT-3.5 for each class.

    This is a placeholder function. In practice, you would implement the
    actual API calls to OpenAI's GPT-3.5 here, using the class names and
    templates as input.

    Args:
        dataset_name: 'cifar10_lt', 'cifar100_lt', 'imagenet_lt'.
        n_prompts_per_class: Number of prompts per class.
        seed: Random seed.
    Returns:
        Dict[int, List[str]]
    """
    # For demonstration, we'll just call get_all_prompts, but in a real
    # implementation, this would involve API calls to GPT-3.5.
    return get_all_prompts(dataset_name, n_prompts_per_class, seed)