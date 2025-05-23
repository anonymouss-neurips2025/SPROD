import os
import random
from PIL import Image
from glob import glob
from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class SpuriousImagenetDataset(Dataset):
    def __init__(self, split, root_dir, transform, is_ood=False, val_count=5, test_count=15,
                 low_shot=False, min_samples=1):
        self.split = split  
        self.root_dir = root_dir
        self.transform = transform
        self.is_ood = is_ood
        self.low_shot = low_shot
        self.min_samples = min_samples
        self.samples = []

        if self.is_ood:
            class_root = os.path.join(root_dir, 'OOD_classes')
            self._load_ood_data(class_root)
        else:
            class_root = os.path.join(root_dir, 'ID_classes')
            self._load_id_data(class_root, val_count, test_count)

    def _load_id_data(self, class_root, val_count, test_count):
        class_folders = sorted(os.listdir(class_root))
        for class_idx, class_name in enumerate(class_folders):
            image_paths = glob(os.path.join(class_root, class_name, '*'))

            rng = random.Random(42)
            rng.shuffle(image_paths)

            if self.split == 'train':
                if self.low_shot:
                    selected = image_paths[:self.min_samples]
                else:
                    selected = image_paths[val_count + test_count:]
            elif self.split == 'val':
                selected = image_paths[:val_count]
            elif self.split == 'test':
                selected = image_paths[val_count:val_count + test_count]
            else:
                raise ValueError(f"Unknown split: {self.split}")

            for img_path in selected:
                self.samples.append((img_path, class_idx))

    def _load_ood_data(self, class_root):
        class_folders = sorted(os.listdir(class_root))
        for class_idx, class_name in enumerate(class_folders):
            image_paths = glob(os.path.join(class_root, class_name, '*'))

            rng = random.Random(42)
            rng.shuffle(image_paths)

            for img_path in image_paths:
                self.samples.append((img_path, class_idx))  

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        return image, label



def get_transform_imagenet(train):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])


def get_spurious_imagenet_loader(split, batch_size, root_dir, is_ood=False, shuffle=False,
                                  val_count=5, test_count=15, 
                                  low_shot=False, min_samples=1):
    transform = get_transform_imagenet(train=False)
    dataset = SpuriousImagenetDataset(
        split=split,
        root_dir=root_dir,
        transform=transform,
        is_ood=is_ood,
        val_count=val_count,
        test_count=test_count,
        low_shot=low_shot,
        min_samples=min_samples
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=2,
        pin_memory=True
    )
    return dataloader
