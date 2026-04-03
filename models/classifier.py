"""Model factory: create model by name + config."""

from .resnet_cifar import resnet20, resnet32, resnet44, resnet56
from .resnet import resnet10, resnet50, resnet101, resnet152
from .resnext import resnext50_32x4d


MODEL_REGISTRY = {
    # CIFAR-scale
    'resnet20': resnet20,
    'resnet32': resnet32,
    'resnet44': resnet44,
    'resnet56': resnet56,
    # ImageNet-scale
    'resnet10': resnet10,
    'resnet50': resnet50,
    'resnet101': resnet101,
    'resnet152': resnet152,
    'resnext50': resnext50_32x4d,
}


def create_model(arch, num_classes, use_norm=False, pretrained=False):
    """Create a model by architecture name.

    Args:
        arch: Architecture name (e.g., 'resnet32', 'resnet50', 'resnext50').
        num_classes: Number of output classes.
        use_norm: If True, use cosine classifier.
        pretrained: If True, load pretrained weights (ImageNet-scale only).

    Returns:
        nn.Module: The model instance.
    """
    if arch not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown architecture: {arch}. "
            f"Supported: {list(MODEL_REGISTRY.keys())}"
        )

    fn = MODEL_REGISTRY[arch]

    # CIFAR-scale models don't support pretrained
    if arch in ('resnet20', 'resnet32', 'resnet44', 'resnet56'):
        return fn(num_classes=num_classes, use_norm=use_norm)
    else:
        return fn(num_classes=num_classes, use_norm=use_norm, pretrained=pretrained)
