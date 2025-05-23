import os
import random
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms


class WaterbirdDataset(Dataset):
    def __init__(self, split, path, transform):
        self.split_map = {'train': 0, 'val': 1, 'test': 2}
        self.env_map = {
            (0, 0): 0,
            (0, 1): 1,
            (1, 0): 2,
            (1, 1): 3
        }

        self.split = split
        self.dataset_dir = path
        self.transform = transform

        if not os.path.exists(self.dataset_dir):
            raise FileNotFoundError(f"Dataset path {self.dataset_dir} does not exist.")

        metadata_path = os.path.join(self.dataset_dir, 'metadata.csv')
        self.metadata_df = pd.read_csv(metadata_path)
        self.metadata_df = self.metadata_df[self.metadata_df['split'] == self.split_map[self.split]]

        self.y_array = self.metadata_df['y'].values
        self.place_array = self.metadata_df['place'].values
        self.filename_array = self.metadata_df['img_filename'].values

    def __len__(self):
        return len(self.filename_array)

    def __getitem__(self, idx):
        y = self.y_array[idx]
        place = self.place_array[idx]
        img_path = os.path.join(self.dataset_dir, self.filename_array[idx])

        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)

        env_label = self.env_map[(y, place)]
        return img, y, env_label
    

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def get_transform_cub(train, seed=None):
    if seed is not None:
        set_seed(seed)

    scale = 256.0 / 224.0
    target_resolution = (224, 224)
    resize_size = (int(target_resolution[0] * scale), int(target_resolution[1] * scale))

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    if train:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(
                target_resolution,
                scale=(0.7, 1.0),
                ratio=(0.75, 1.3333333333333333),
                interpolation=2
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(resize_size),
            transforms.CenterCrop(target_resolution),
            transforms.ToTensor(),
            normalize
        ])

    return transform


def get_waterbird_dataloader(split, transform, path, batch_size):
    kwargs = {'pin_memory': True, 'num_workers': 2, 'drop_last': False}
    dataset = WaterbirdDataset(split=split, path=path, transform=transform)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        **kwargs
    )
    return dataloader


def get_waterbird_loaders(path, batch_size, use_train_transform=False, seed=None):
    t_train = get_transform_cub(train=True, seed=seed)
    t_test = get_transform_cub(train=False, seed=seed)

    transform_train = t_train if use_train_transform else t_test

    trainloader = get_waterbird_dataloader('train', transform_train, path, batch_size)
    valloader = get_waterbird_dataloader('val', t_test, path, batch_size)
    testloader = get_waterbird_dataloader('test', t_test, path, batch_size)

    return trainloader, valloader, testloader

def get_waterbird_dataset(split, path, transform):
    return WaterbirdDataset(split=split, path = path, transform = transform)


def get_waterbird_dataloader_for_env(path, batch_size, target_env):
    transform = get_transform_cub(train=False)
    dataset = WaterbirdDataset(split='test', path=path, transform=transform)

    # Filter indices based on the desired environment condition
    env_indices = [
        idx for idx in range(len(dataset))
        if dataset.env_dict[(dataset.y_array[idx], dataset.place_array[idx])] == target_env
    ]

    filtered_dataset = Subset(dataset, env_indices)

    dataloader = DataLoader(
        dataset=filtered_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=False
    )

    return dataloader