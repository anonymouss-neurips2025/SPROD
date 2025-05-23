from torch.utils.data import Dataset, DataLoader
import numpy as np

class FeatDataset(Dataset):
    def __init__(self, feat, labels):
        self.data = feat
        self.labels = labels
        self.len = feat.shape[0]
        assert self.len == len(labels)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.labels[idx]
        return data, label



class FeatDatasetUrb(Dataset):
    def __init__(self, embeddings, labels,places,co):
        self.embeddings = embeddings
        self.labels = labels
        self.places = places
        self.co = co

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx], self.places[idx], self.co[idx]


def load_embeddings(path):
    return np.load(path, allow_pickle=True).item()


def parse_split(value):
    return value if isinstance(value, str) else {0: 'train', 1: 'val', 2: 'test'}.get(value, 'unknown')


def load_embeddings_and_labels(path):
    emb_dict = load_embeddings(path)
    splits = {'0': ([], []), '1': ([], []), '2': ([], [])}

    for name, emb in emb_dict.items():
        label, _, split = name.split('_')[:3]
        split = parse_split(split)
        if split in splits:
            splits[split][0].append(emb)
            splits[split][1].append(int(label))
    return tuple(map(lambda x: (np.array(x[0]), np.array(x[1])), splits.values()))


def load_embeddings_and_labels_urb(path):
    emb_dict = load_embeddings(path)
    splits = {'0': ([], [], [], []), '1': ([], [], [], []), '2': ([], [], [], [])}

    for name, emb in emb_dict.items():
        label, place, co, split = name.split('_')[:4]
        split = parse_split(split)
        if split in splits:
            emb_list, lbl_list, plc_list, co_list = splits[split]
            emb_list.append(emb)
            lbl_list.append(int(label))
            plc_list.append(int(place))
            co_list.append(int(co))

    return tuple(map(lambda x: tuple(map(np.array, x)), splits.values()))

def create_dataloaders(train_data, val_data, test_data, batch_size, is_urb=False):
    DatasetClass = FeatDatasetUrb if is_urb else FeatDataset
    train_loader = DataLoader(DatasetClass(*train_data), batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(DatasetClass(*val_data), batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(DatasetClass(*test_data), batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, test_loader

