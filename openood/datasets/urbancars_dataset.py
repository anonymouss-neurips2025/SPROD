import os
import torch
import random
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader



def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True



class UrbanCarsDataset(Dataset):
    def __init__(self, transform, train_dir, valtest_dir, split='train'):
        self.class_dict = {'country': 0, 'urban': 1}
        self.transform = transform
        self.split = split

        subdir_map = {
            'train': os.path.join(train_dir, 'bg-0.95_co_occur_obj-0.95/train'),
            'val': os.path.join(valtest_dir, 'bg-0.5_co_occur_obj-0.5/val'),
            'test': os.path.join(valtest_dir, 'bg-0.5_co_occur_obj-0.5/test'),
        }
        self.dataset_dir = subdir_map[split]
        if not self.dataset_dir or not os.path.exists(self.dataset_dir):
            raise ValueError(f'Dataset path does not exist: {self.dataset_dir}')

        self.image_paths = []
        self.y_array = []
        self.place_array = []
        self.co_occ_array = []

        for folder_name in os.listdir(self.dataset_dir):
            folder_path = os.path.join(self.dataset_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue

            try:
                _, class_info, place_info, co_occ_name = folder_name.split('-')
                class_name = class_info.split('_')[0]
                place_name = place_info.split('_')[0]
            except ValueError:
                continue  

            class_idx = self.class_dict[class_name]
            if class_idx is None:
                continue

            for img_name in os.listdir(folder_path):
                if img_name.endswith('.jpg') and len(img_name.split('_')) == 1:
                    img_path = os.path.join(folder_path, img_name)
                    self.image_paths.append(img_path)
                    self.y_array.append(class_idx)
                    self.place_array.append(place_name)
                    self.co_occ_array.append(co_occ_name)

        self.y_array = torch.tensor(self.y_array, dtype=torch.long)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.y_array[idx]
        place = self.place_array[idx]
        co_occ = self.co_occ_array[idx]

        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)

        return img, label, place, co_occ

    def get_raw_image(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        preprocess = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
        ])
        return preprocess(img)


def get_transform_cub(train, seed=None):
    if seed is not None:
        set_seed(seed)

    scale = 256.0 / 224.0
    target_resolution = [224, 224]
    assert target_resolution is not None

    if not train:
        transform = transforms.Compose([
            transforms.Resize((int(target_resolution[0] * scale), int(target_resolution[1] * scale))),
            transforms.CenterCrop(target_resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    else:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(target_resolution, scale=(0.7, 1.0), ratio=(0.75, 1.33), interpolation=2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    return transform



class JustBGOOD(Dataset):
    def __init__(self, transform, root_dir):
        self.transform = transform
        self.image_paths = []
        self.groups = []

        self.env_dict = {
            'country_country': 0,
            'country_urban': 1,
            'urban_country': 2,
            'urban_urban': 3
        }

        data_dir = os.path.join(root_dir, 'bg-0.5_co_occur_obj-0.5')

        for folder_name in os.listdir(data_dir):
            folder_path = os.path.join(data_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue

            for root, _, files in os.walk(folder_path):
                parts = root.split(os.sep)

                if len(parts) <= 6:
                    continue

                try:
                    _, _, place_info, co_occ_name = parts[6].split('-')
                    place_name = place_info.split('_')[0]
                    group_key = f'{place_name}_{co_occ_name}'
                    group_id = self.env_dict[group_key]
                except (ValueError, KeyError):
                    continue  

                for img_name in files:
                    if img_name.endswith('.jpg') and len(img_name.split('_')) == 3 and img_name.split('_')[1] == 'just':
                        img_path = os.path.join(root, img_name)
                        self.image_paths.append(img_path)
                        self.groups.append(group_id)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        return img, self.groups[idx]


class NoCarOOD(Dataset):
    def __init__(self, transform, root_dir):
        self.transform = transform
        self.image_paths = []
        self.groups = []

        self.env_dict = {
            'country_country': 0,
            'country_urban': 1,
            'urban_country': 2,
            'urban_urban': 3
        }

        data_dir = os.path.join(root_dir, 'bg-0.5_co_occur_obj-0.5')

        for folder_name in os.listdir(data_dir):
            folder_path = os.path.join(data_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue

            for root, _, files in os.walk(folder_path):
                parts = root.split(os.sep)

                if len(parts) <= 6:
                    continue

                try:
                    _, _, place_info, co_occ_name = parts[6].split('-')
                    place_name = place_info.split('_')[0]
                    group_key = f'{place_name}_{co_occ_name}'
                    group_id = self.env_dict[group_key]
                except (ValueError, KeyError):
                    continue  

                for img_name in files:
                    if img_name.endswith('.jpg') and len(img_name.split('_')) == 3 and img_name.split('_')[1] == 'no':
                        img_path = os.path.join(root, img_name)
                        self.image_paths.append(img_path)
                        self.groups.append(group_id)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        return img, self.groups[idx]
    



def get_urbancars_dataloader(split, batch_size, train_dir, valtest_dir):
    t_test = get_transform_cub(train=False)
    dataset = UrbanCarsDataset(
        transform=t_test,
        split=split,
        train_dir=train_dir,
        valtest_dir=valtest_dir
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=2,
        drop_last=False
    )


def get_urbancars_loaders(batch_size, train_dir, valtest_dir, use_train_transform=False, seed=None):
    t_train = get_transform_cub(train=True, seed=seed)
    t_test = get_transform_cub(train=False, seed=seed)

    train_transform = t_train if use_train_transform else t_test

    train_loader = get_urbancars_dataloader('train', train_transform, batch_size, train_dir, valtest_dir)
    val_loader = get_urbancars_dataloader('val', t_test, batch_size, train_dir, valtest_dir)
    test_loader = get_urbancars_dataloader('test', t_test, batch_size, train_dir, valtest_dir)

    return train_loader, val_loader, test_loader


def get_urbancars_justbgood_dataloader(args):
    dataset = JustBGOOD(get_transform_cub(train=False))
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=8,
        drop_last=True
    )


def get_urbancars_nocarood_dataloader(args):
    dataset = NoCarOOD(get_transform_cub(train=False))
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=8,
        drop_last=True
    )