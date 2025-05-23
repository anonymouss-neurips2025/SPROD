import os
import torch
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import random


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class celebABlondDataset(Dataset):
    def __init__(self, split, seed, path, data_label_correlation, transform, balance_correlation_sizes = False):
        self.split_dict = {
            'train': 0,
            'val': 1,
            'test': 2
        }

        self.env_dict = {
            (0, 0): 0,  # non-blond hair, female
            (0, 1): 1,  # non-blond hair, male
            (1, 0): 2,  # blond hair, female
            (1, 1): 3   # blond hair, male
        }

        self.split = split
        self.seed = seed
        self.dataset_dir = path
        if not os.path.exists(self.dataset_dir):
            raise ValueError(f'{self.dataset_dir} does not exist. Please generate the dataset first.')

        self.metadata_df = pd.read_csv(os.path.join(self.dataset_dir, 'list_attr_celeba.csv'))
        split_df = pd.read_csv(os.path.join(self.dataset_dir, 'list_eval_partition.csv'))
        self.metadata_df = self.metadata_df.merge(split_df, on="image_id")

        self.metadata_df = self.metadata_df[self.metadata_df['partition'] == self.split_dict[self.split]]

        self.y_array = np.array([0 if el == -1 else el for el in self.metadata_df['Blond_Hair'].values])
        self.gender_array = np.array([0 if el == -1 else el for el in self.metadata_df['Male'].values])
        self.filename_array = self.metadata_df['image_id'].values

        self.transform = transform

        self.balance_correlation_sizes = balance_correlation_sizes

        if self.split == 'train':
            self.subsample(data_label_correlation)
        elif self.split in ['test', 'val']:
            self.subsample(.5)

    def subsample(self, ratio = 0.5):
        set_seed(self.seed)

        train_group_idx = {(0, 0): [], (0, 1): [], (1, 0): [], (1, 1): []}

        for idx, (y, gender) in enumerate(zip(self.y_array, self.gender_array)):
            train_group_idx[(y, gender)].append(idx)
        
        small_sample_size = len(train_group_idx[(1, 1)])
        big_sample_size = int(ratio/(1-ratio)*len(train_group_idx[(1, 1)]))

        undersampled_idx_00 = np.random.choice(train_group_idx[(0, 0)], small_sample_size, replace=False)
        undersampled_idx_11 = train_group_idx[(1, 1)]
        undersampled_idx_01 = np.random.choice(train_group_idx[(0, 1)], big_sample_size, replace=False)
        undersampled_idx_10 = np.random.choice(train_group_idx[(1, 0)], big_sample_size, replace=False)
        
        undersampled_idx = np.concatenate( (undersampled_idx_00, undersampled_idx_11, undersampled_idx_01, undersampled_idx_10) )
        undersampled_idx = undersampled_idx.astype(int)

        self.y_array = self.y_array[undersampled_idx]
        self.gender_array = self.gender_array[undersampled_idx]
        self.filename_array = self.filename_array[undersampled_idx]


        if self.balance_correlation_sizes:
            size_map = {"train": 5548, "val": 728, "test": 720}
            size = size_map.get(self.split, len(self.y_array))
            subsetted_idx = np.random.choice(len(self.y_array), size, replace=False)

            self.y_array = self.y_array[subsetted_idx]
            self.gender_array = self.gender_array[subsetted_idx]
            self.filename_array = self.filename_array[subsetted_idx]

    def __len__(self):
        return len(self.filename_array)

    def __getitem__(self, idx):
        y = self.y_array[idx]
        gender = self.gender_array[idx]
        img_filename = os.path.join(
            self.dataset_dir,
            'img_align_celeba',
            self.filename_array[idx])
        img = Image.open(img_filename).convert('RGB')
        img = self.transform(img)
        return img, y, self.env_dict[(y, gender)]
    


class celebAOodDataset(Dataset):
    def __init__(self, path):
        self.dataset_dir = path
        if not os.path.exists(self.dataset_dir):
            raise ValueError(
                f'{self.dataset_dir} does not exist yet. Please generate the dataset first.')
        self.metadata_df = pd.read_csv(
            os.path.join(self.dataset_dir, 'celebA_ood.csv'))

        self.filename_array = self.metadata_df['image_id'].values
        self.transform = get_transform_celeba(train=False)

    def __len__(self):
        return len(self.filename_array)

    def __getitem__(self, idx):
        img_filename = os.path.join(
            self.dataset_dir,
            'img_align_celeba',
            self.filename_array[idx])
        img = Image.open(img_filename).convert('RGB')
        img = self.transform(img)

        return img

def get_transform_celeba(train, seed=None):
    if seed is not None:
        set_seed(seed)

    orig_w = 178
    orig_h = 218
    orig_min_dim = min(orig_w, orig_h)
    target_resolution = (224, 224)

    if not train:
        transform = transforms.Compose([
            transforms.CenterCrop(orig_min_dim),
            transforms.Resize(target_resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(
                target_resolution,
                scale=(0.7, 1.0),
                ratio=(1.0, 1.3333333333333333),
                interpolation=2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    return transform




def get_celebA_blond_ood_dataloader(ood_batch_size, path):
    dataset = celebAOodDataset(path=path)
    return DataLoader(
        dataset=dataset,
        batch_size=ood_batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=8,
        drop_last=True
    )



def get_celebA_dataloader(split, path, batch_size, correlation, seed):
    transform = get_transform_celeba(split=='train', seed=seed)
    dataset = celebABlondDataset(
        split=split,
        seed=seed,
        path=path,
        data_label_correlation=correlation,
        transform=transform

    )
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        pin_memory=True,
        num_workers=2,
        drop_last=False
    )


def get_celebA_blond_loaders(batch_size, correlation, seed, path):
    train_loader = get_celebA_dataloader('train', path, batch_size, correlation, seed)
    val_loader = get_celebA_dataloader('val', path, batch_size, correlation, seed)
    test_loader = get_celebA_dataloader('test', path, batch_size, correlation, seed)
    
    return train_loader, val_loader, test_loader

