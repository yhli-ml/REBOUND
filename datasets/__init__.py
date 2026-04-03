from .cifar_lt import IMBALANCECIFAR10, IMBALANCECIFAR100
from .imagenet_lt import ImageNetLT
from .inaturalist import iNaturalist
from .places_lt import PlacesLT

__all__ = [
    'IMBALANCECIFAR10', 'IMBALANCECIFAR100',
    'ImageNetLT', 'iNaturalist', 'PlacesLT',
    'get_dataset',
]


def get_dataset(name, root, train=True, transform=None, imb_factor=0.01,
                download=False, img_root=None):
    """Factory function to get a long-tail dataset by name.

    Args:
        name: Dataset name (cifar10_lt, cifar100_lt, imagenet_lt, inaturalist, places_lt)
        root: Root directory for the dataset (or annotation txt directory)
        train: Whether to get train or test split
        transform: Data transforms
        imb_factor: Imbalance factor (for CIFAR only)
        download: Whether to download (for CIFAR only)
        img_root: Root directory for images (for ImageNet-LT/iNaturalist/Places-LT
                  where images and annotations may be in different locations).
                  If None, images are loaded relative to `root`.

    Returns:
        dataset: Dataset instance with `cls_num_list` attribute
    """
    name = name.lower()
    if name == 'cifar10_lt':
        return IMBALANCECIFAR10(root=root, train=train, transform=transform,
                                imb_factor=imb_factor, download=download)
    elif name == 'cifar100_lt':
        return IMBALANCECIFAR100(root=root, train=train, transform=transform,
                                 imb_factor=imb_factor, download=download)
    elif name == 'imagenet_lt':
        split = 'train' if train else 'val'
        return ImageNetLT(root=root, split=split, transform=transform, img_root=img_root)
    elif name == 'inaturalist':
        split = 'train' if train else 'val'
        return iNaturalist(root=root, split=split, transform=transform, img_root=img_root)
    elif name == 'places_lt':
        split = 'train' if train else 'val'
        return PlacesLT(root=root, split=split, transform=transform, img_root=img_root)
    else:
        raise ValueError(f"Unknown dataset: {name}. "
                         f"Supported: cifar10_lt, cifar100_lt, imagenet_lt, inaturalist, places_lt")
