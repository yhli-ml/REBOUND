"""ResNeXt-50 (32x4d) for ImageNet-scale inputs."""

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


class ResNeXt(nn.Module):
    """ResNeXt wrapper for long-tail learning.

    Args:
        num_classes: Number of output classes.
        use_norm: If True, use cosine classifier.
        pretrained: If True, load ImageNet pretrained weights.
    """

    def __init__(self, num_classes=1000, use_norm=False, pretrained=False):
        super().__init__()

        weights = 'IMAGENET1K_V1' if pretrained else None
        backbone = tv_models.resnext50_32x4d(weights=weights)

        self.feat_dim = backbone.fc.in_features

        self.encoder = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4,
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)

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
        self.num_classes = num_classes
        if use_norm:
            self.fc = NormedLinear(self.feat_dim, num_classes)
        else:
            self.fc = nn.Linear(self.feat_dim, num_classes)


def resnext50_32x4d(num_classes=1000, use_norm=False, pretrained=False):
    return ResNeXt(num_classes, use_norm, pretrained)
