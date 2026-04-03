"""ResNet for ImageNet-scale (224x224 input).

Uses torchvision's ResNet implementation with optional modifications
for long-tail learning (cosine classifier, feature extraction).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models


class NormedLinear(nn.Module):
    """Cosine classifier."""

    def __init__(self, in_features, out_features, scale=30.0):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.scale = scale
        nn.init.kaiming_uniform_(self.weight, a=1)

    def forward(self, x):
        x = F.normalize(x, dim=1)
        w = F.normalize(self.weight, dim=1)
        return self.scale * F.linear(x, w)


class ResNet(nn.Module):
    """ResNet wrapper for long-tail learning.

    Args:
        arch: Architecture name ('resnet50', 'resnet101', 'resnet152').
        num_classes: Number of output classes.
        use_norm: If True, use cosine classifier.
        pretrained: If True, load ImageNet pretrained weights.
    """

    def __init__(self, arch='resnet50', num_classes=1000,
                 use_norm=False, pretrained=False):
        super().__init__()

        # Load torchvision ResNet
        if pretrained:
            weights = 'IMAGENET1K_V1'
        else:
            weights = None

        if arch == 'resnet50':
            backbone = tv_models.resnet50(weights=weights)
        elif arch == 'resnet101':
            backbone = tv_models.resnet101(weights=weights)
        elif arch == 'resnet152':
            backbone = tv_models.resnet152(weights=weights)
        elif arch == 'resnet10':
            backbone = _resnet10()
        else:
            raise ValueError(f"Unknown arch: {arch}")

        self.feat_dim = backbone.fc.in_features

        # Remove the original fc layer
        self.encoder = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4,
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # Classification head
        self.num_classes = num_classes
        if num_classes > 0:
            if use_norm:
                self.fc = NormedLinear(self.feat_dim, num_classes)
            else:
                self.fc = nn.Linear(self.feat_dim, num_classes)
        else:
            self.fc = None

    def forward(self, x, return_feat=False):
        feat = self.encoder(x)
        feat = self.avgpool(feat)
        feat = feat.view(feat.size(0), -1)

        if self.fc is not None:
            logits = self.fc(feat)
        else:
            logits = feat

        if return_feat:
            return logits, feat
        return logits

    def get_classifier(self):
        return self.fc

    def reset_classifier(self, num_classes, use_norm=False):
        """Reset classifier head (for cRT / two-stage methods)."""
        self.num_classes = num_classes
        if use_norm:
            self.fc = NormedLinear(self.feat_dim, num_classes)
        else:
            self.fc = nn.Linear(self.feat_dim, num_classes)


def _resnet10():
    """Build a ResNet-10 (not in torchvision by default)."""
    return tv_models.ResNet(
        tv_models.resnet.BasicBlock, [1, 1, 1, 1]
    )


def resnet10(num_classes=1000, use_norm=False, pretrained=False):
    return ResNet('resnet10', num_classes, use_norm, pretrained=False)


def resnet50(num_classes=1000, use_norm=False, pretrained=False):
    return ResNet('resnet50', num_classes, use_norm, pretrained)


def resnet101(num_classes=1000, use_norm=False, pretrained=False):
    return ResNet('resnet101', num_classes, use_norm, pretrained)


def resnet152(num_classes=1000, use_norm=False, pretrained=False):
    return ResNet('resnet152', num_classes, use_norm, pretrained)
