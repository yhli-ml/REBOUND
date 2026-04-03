"""iNaturalist 2018 dataset.

Expects the following file structure:
    root/
        train/
        val/
        train2018.json   (or iNaturalist18_train.txt)
        val2018.json     (or iNaturalist18_val.txt)

Supports both JSON (official) and txt annotation formats.
"""

import json
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class iNaturalist(Dataset):
    """iNaturalist 2018 dataset (8142 classes, naturally long-tailed).

    Args:
        root: Root directory containing images and annotation files.
        split: 'train' or 'val'.
        transform: Optional transform to apply to images.
    """

    num_classes = 8142

    def __init__(self, root, split='train', transform=None, img_root=None):
        self.root = root
        self.img_root = img_root if img_root is not None else root
        self.split = split
        self.transform = transform

        self.img_paths = []
        self.targets = []

        # Try txt format first (simpler)
        txt_file = os.path.join(root, f'iNaturalist18_{split}.txt')
        json_file = os.path.join(root, f'{split}2018.json')

        if os.path.exists(txt_file):
            self._load_txt(txt_file)
        elif os.path.exists(json_file):
            self._load_json(json_file)
        else:
            raise FileNotFoundError(
                f"Neither {txt_file} nor {json_file} found.\n"
                f"Please download iNaturalist 2018 from: "
                f"https://github.com/visipedia/inat_comp/tree/master/2018"
            )

        # Compute class counts
        self.cls_num_list = [0] * self.num_classes
        for t in self.targets:
            self.cls_num_list[t] += 1

    def _load_txt(self, txt_file):
        with open(txt_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                self.img_paths.append(os.path.join(self.img_root, parts[0]))
                self.targets.append(int(parts[1]))

    def _load_json(self, json_file):
        with open(json_file, 'r') as f:
            data = json.load(f)
        id_to_label = {cat['id']: idx for idx, cat in enumerate(data['categories'])}
        ann_map = {ann['image_id']: ann['category_id'] for ann in data['annotations']}
        for img_info in data['images']:
            self.img_paths.append(os.path.join(self.img_root, img_info['file_name']))
            self.targets.append(id_to_label[ann_map[img_info['id']]])

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
