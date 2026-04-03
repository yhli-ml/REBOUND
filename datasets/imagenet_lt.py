"""ImageNet-LT dataset.

Expects the following structure:
    txt_root/                          (e.g., data/ImageNet_LT/)
        ImageNet_LT_train.txt
        ImageNet_LT_val.txt
        ImageNet_LT_test.txt
    img_root/                          (e.g., /hss/giil/temp/data/imagenet)
        train/n01440764/xxx.JPEG
        val/n01440764/xxx.JPEG

Each txt file has lines: "relative_path label"
  e.g., train/n01440764/n01440764_190.JPEG 0
"""

import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class ImageNetLT(Dataset):
    """ImageNet-LT dataset (1000 classes, long-tailed training split).

    Args:
        root: Directory containing annotation txt files.
        split: 'train', 'val', or 'test'.
        transform: Optional transform to apply to images.
        img_root: Root directory containing the actual images (train/, val/).
                  If None, image paths in txt are treated as relative to `root`.
    """

    num_classes = 1000

    def __init__(self, root, split='train', transform=None, img_root=None):
        self.root = root
        self.img_root = img_root if img_root is not None else root
        self.split = split
        self.transform = transform

        # Load annotations
        txt_file = os.path.join(root, f'ImageNet_LT_{split}.txt')
        if not os.path.exists(txt_file):
            raise FileNotFoundError(
                f"Annotation file not found: {txt_file}\n"
                f"Please download ImageNet-LT annotations from: "
                f"https://github.com/zhmiao/OpenLongTailRecognition-OLTR"
            )

        self.img_paths = []
        self.targets = []
        with open(txt_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                self.img_paths.append(os.path.join(self.img_root, parts[0]))
                self.targets.append(int(parts[1]))

        # Compute class counts
        self.cls_num_list = [0] * self.num_classes
        for t in self.targets:
            self.cls_num_list[t] += 1

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        path = self.img_paths[index]
        target = self.targets[index]
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def get_cls_num_list(self):
        return self.cls_num_list


def get_imagenet_train_transform(augment='standard'):
    """Get training transform for ImageNet-scale datasets."""
    transform_list = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
    if augment == 'autoaug':
        transform_list.append(transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET))
    elif augment == 'randaug':
        transform_list.append(transforms.RandAugment(num_ops=2, magnitude=9))

    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return transforms.Compose(transform_list)


def get_imagenet_test_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
