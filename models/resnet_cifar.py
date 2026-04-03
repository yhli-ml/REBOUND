"""ResNet for CIFAR (32x32 input).

Based on "Deep Residual Learning for Image Recognition" (He et al., 2016).
Architecture follows the original paper: 3 groups of BasicBlocks with
{16, 32, 64} channels, suitable for 32x32 CIFAR images.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet_CIFAR(nn.Module):
    """ResNet for CIFAR-sized (32x32) inputs.

    Args:
        block: Block type (BasicBlock).
        num_blocks: List of 3 ints, number of blocks per group.
        num_classes: Number of output classes (0 = feature extractor only).
        use_norm: If True, use cosine classifier (NormedLinear).
    """

    def __init__(self, block, num_blocks, num_classes=10, use_norm=False):
        super().__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)

        self.feat_dim = 64 * block.expansion
        self.num_classes = num_classes

        if num_classes > 0:
            if use_norm:
                self.fc = NormedLinear(self.feat_dim, num_classes)
            else:
                self.fc = nn.Linear(self.feat_dim, num_classes)
        else:
            self.fc = None

        self._initialize_weights()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, return_feat=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.adaptive_avg_pool2d(out, 1)
        feat = out.view(out.size(0), -1)

        if self.fc is not None:
            logits = self.fc(feat)
        else:
            logits = feat

        if return_feat:
            return logits, feat
        return logits


class NormedLinear(nn.Module):
    """Cosine classifier (normalized weights and features)."""

    def __init__(self, in_features, out_features, scale=30.0):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.scale = scale
        nn.init.kaiming_uniform_(self.weight, a=1)

    def forward(self, x):
        x = F.normalize(x, dim=1)
        w = F.normalize(self.weight, dim=1)
        return self.scale * F.linear(x, w)


def resnet20(num_classes=10, use_norm=False):
    return ResNet_CIFAR(BasicBlock, [3, 3, 3], num_classes=num_classes, use_norm=use_norm)


def resnet32(num_classes=10, use_norm=False):
    return ResNet_CIFAR(BasicBlock, [5, 5, 5], num_classes=num_classes, use_norm=use_norm)


def resnet44(num_classes=10, use_norm=False):
    return ResNet_CIFAR(BasicBlock, [7, 7, 7], num_classes=num_classes, use_norm=use_norm)


def resnet56(num_classes=10, use_norm=False):
    return ResNet_CIFAR(BasicBlock, [9, 9, 9], num_classes=num_classes, use_norm=use_norm)
