import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from pathlib import Path
import yaml

repo_root = Path(__file__).resolve().parents[2]
with open(repo_root / "configs/sprod_paths.yaml", "r") as f:
    config = yaml.safe_load(f)

paths = config["paths"]

dataset_root = paths['ID_data']
checkpoint_root = paths["checkpoints"]
embedding_save_dir = paths["ID_emb"]


class AnimalsEmbeddingDataset(Dataset):
    def __init__(self, model_name, split='train', in_classes=None, low_shot=False, num_min_samples=None,seed=None):
        """
        Args:
            embeddings_path (str): Path to the saved embeddings file (as a .npy file).
            metadata_path (str): Path to the metadata CSV file containing Id, address, class, attribute, and split columns.
            split (str): The data split ('train', 'val', or 'test') to filter metadata.
            in_distribution_classes (list): List of classes to include in the dataset as in-distribution.
        """
        emb_path = f'{embedding_save_dir}/animals/animals_embs_{model_name}_95.npy'
        metadata_path = f'{dataset_root}/animals_metadata.csv'

        self.embeddings = np.load(emb_path, allow_pickle=True).item()
        self.metadata = pd.read_csv(metadata_path)
        self.split = split
        self.in_distribution_classes = in_classes
        split_dict = {'train': 0, 'val': 1, 'test': 2}

        self.metadata = self.metadata[self.metadata['split'] == split_dict[self.split]]
        if self.in_distribution_classes:
            self.metadata = self.metadata[self.metadata['class'].isin(self.in_distribution_classes)]

        if split == 'train' and low_shot and num_min_samples is not None:
            np.random.seed(seed)
            low_shot_meta = []
            for cls in self.metadata['class'].unique():
                cls_df = self.metadata[self.metadata['class'] == cls]
                sampled = cls_df.sample(n=min(num_min_samples, len(cls_df)), replace=False, random_state=seed)
                low_shot_meta.append(sampled)
            self.metadata = pd.concat(low_shot_meta)

        unique_classes = sorted(self.metadata['class'].unique())
        self.class_to_label = {cls: idx for idx, cls in enumerate(unique_classes)}
        

        self.dataset = []
        for _, row in self.metadata.iterrows():

            image_id = row['Id']
            class_label = self.class_to_label[row['class']]
            place = row['attribute'] 

            if image_id in self.embeddings:
                embedding = torch.tensor(self.embeddings[image_id], dtype=torch.float32)
                self.dataset.append((embedding, class_label, place))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        embedding, label, place = self.dataset[idx]
        return embedding, label, place




class AnimalsOODDataset(Dataset):
    def __init__(self, model_name, ood_classes):
        """
        Args:
            embeddings_path (str): Path to the saved embeddings file (as a .npy file).
            metadata_path (str): Path to the metadata CSV file containing Id, address, class, attribute, and split columns.
            ood_classes (list): List of classes to include as out-of-distribution (OOD).
        """
        
        emb_path = f'{embedding_save_dir}/animals/animals_embs_{model_name}_95.npy'
        metadata_path = f'{dataset_root}/animals_metadata.csv'

        self.embeddings = np.load(emb_path, allow_pickle=True).item()
        self.metadata = pd.read_csv(metadata_path)
        self.ood_classes = ood_classes

        self.metadata = self.metadata[self.metadata['class'].isin(self.ood_classes)]

        self.dataset = []
        for _, row in self.metadata.iterrows():
            image_id = row['Id']
            place = row['attribute']  

            if image_id in self.embeddings:
                embedding = torch.tensor(self.embeddings[image_id], dtype=torch.float32)
                self.dataset.append((embedding, -1, place))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        embedding, label, place = self.dataset[idx]
        return embedding, label, place





ALL_CLASSES = [
    'cat', 'cow', 'crab', 'dog', 'dolphin', 'elephant', 'fox', 'frog', 'giraffe',
    'goose', 'horse', 'lion', 'monkey', 'owl', 'rabbit', 'rat', 'seal', 'sheep',
    'squirrel', 'tiger', 'wolf', 'bear', 'kangaroo', 'lizard', 'ostrich', 'tortoise'
]

def get_animal_dataloader(model_name, split, in_classes, batch_size, shuffle=True, low_shot=False, num_min_samples=None, seed = None):
    """
    Returns a DataLoader for in-distribution data based on the specified split.
    """

    kwargs = {'pin_memory': True, 'num_workers': 0, 'drop_last': False}
    dataset = AnimalsEmbeddingDataset(
        model_name=model_name,
        split=split,
        in_classes=in_classes,
        low_shot=low_shot,
        num_min_samples=num_min_samples,
        seed=seed
    )
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)
    return dataloader

def get_animalood_dataloader(model_name, ood_classes, batch_size, shuffle=True):
    """
    Returns a DataLoader for out-of-distribution data.
    """

    kwargs = {'pin_memory': True, 'num_workers': 0, 'drop_last': True}
    dataset = AnimalsOODDataset(
        model_name=model_name,
        ood_classes=ood_classes
    )
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)
    return dataloader

def get_animal_loaders(model_name, batch_size, ood_classes,low_shot=False, num_min_samples=None, seed=None):
    """
    Returns DataLoaders for train, val, test splits of in-distribution data and a loader for OOD data.
    
    Args:
        embeddings_path (str): Path to the saved embeddings file (as a .npy file).
        metadata_path (str): Path to the metadata CSV file.
        batch_size (int): Batch size for the loaders.
        ood_classes (list): List of classes to include as OOD.

    Returns:
        trainloader, valloader, testloader, oodloader: DataLoaders for in-distribution and OOD data.
    """
    in_classes = [cls for cls in ALL_CLASSES if cls not in ood_classes]

    trainloader = get_animal_dataloader(model_name, 'train', in_classes, batch_size,low_shot=low_shot, num_min_samples=num_min_samples, seed=seed)
    valloader = get_animal_dataloader(model_name, 'val', in_classes, batch_size)
    testloader = get_animal_dataloader(model_name, 'test', in_classes, batch_size)

    oodloader = get_animalood_dataloader(model_name, ood_classes, batch_size)

    return trainloader, valloader, testloader, oodloader
