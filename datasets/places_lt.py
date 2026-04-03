"""Places-LT dataset.

Expects the following file structure:
    root/
        data/vision/torchvision/datasets/places365/  (or images/)
        Places_LT_train.txt
        Places_LT_val.txt
        Places_LT_test.txt

Each txt file has lines: "path label"
"""

import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class PlacesLT(Dataset):
    """Places-LT dataset (365 classes, long-tailed training split).

    Args:
        root: Root directory containing images and annotation txt files.
        split: 'train', 'val', or 'test'.
        transform: Optional transform to apply to images.
    """

    num_classes = 365

    def __init__(self, root, split='train', transform=None, img_root=None):
        self.root = root
        self.img_root = img_root if img_root is not None else root
        self.split = split
        self.transform = transform

        txt_file = os.path.join(root, f'Places_LT_{split}.txt')
        if not os.path.exists(txt_file):
            raise FileNotFoundError(
                f"Annotation file not found: {txt_file}\n"
                f"Please download Places-LT annotations from: "
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
