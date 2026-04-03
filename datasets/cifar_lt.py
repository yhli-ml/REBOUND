"""CIFAR-10-LT and CIFAR-100-LT datasets with controllable imbalance."""

import numpy as np
import torchvision
from torchvision import transforms


class IMBALANCECIFAR10(torchvision.datasets.CIFAR10):
    """CIFAR-10 with long-tailed (exponential) imbalance.

    Args:
        root: Root directory for CIFAR-10 data.
        train: If True, use training set.
        imb_factor: Imbalance factor (ratio of min class to max class count).
                    1.0 = balanced, 0.01 = 100x imbalance.
        transform: Data transform pipeline.
        download: If True, download the dataset.
    """

    num_classes = 10

    def __init__(self, root, train=True, imb_factor=0.01,
                 transform=None, target_transform=None, download=False):
        super().__init__(root, train=train, transform=transform,
                         target_transform=target_transform, download=download)
        self.imb_factor = imb_factor
        if train:
            self.cls_num_list = self._get_img_num_per_cls(self.num_classes, imb_factor)
            self._gen_imbalanced_data(self.cls_num_list)
        else:
            self.cls_num_list = None

    def _get_img_num_per_cls(self, num_classes, imb_factor):
        """Get number of images per class with exponential decay."""
        img_max = len(self.data) / num_classes
        cls_num_list = []
        for cls_idx in range(num_classes):
            num = img_max * (imb_factor ** (cls_idx / (num_classes - 1.0)))
            cls_num_list.append(int(num))
        return cls_num_list

    def _gen_imbalanced_data(self, cls_num_list):
        """Subsample the dataset to create imbalanced distribution."""
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)

        # Deterministic for reproducibility
        np.random.seed(0)
        for the_class, the_img_num in zip(classes, cls_num_list):
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class] * the_img_num)

        self.data = np.vstack(new_data)
        self.targets = new_targets

    def get_cls_num_list(self):
        return self.cls_num_list


class IMBALANCECIFAR100(IMBALANCECIFAR10):
    """CIFAR-100 with long-tailed (exponential) imbalance."""

    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]
    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }
    num_classes = 100


# Default transforms for CIFAR
def get_cifar_train_transform(augment='standard'):
    """Get training transform for CIFAR datasets.

    Args:
        augment: 'standard', 'autoaug', or 'randaug'
    """
    transform_list = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ]
    if augment == 'autoaug':
        transform_list.append(transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10))
    elif augment == 'randaug':
        transform_list.append(transforms.RandAugment(num_ops=2, magnitude=9))

    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    return transforms.Compose(transform_list)


def get_cifar_test_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
