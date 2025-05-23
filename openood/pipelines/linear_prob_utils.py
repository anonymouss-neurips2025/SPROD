import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import random
import os

from openood.datasets.feature_dataset import load_embeddings_and_labels, load_embeddings_and_labels_urb, create_dataloaders
from openood.datasets.animals_metacoco_dataset import get_animal_loaders


# --- Model head ---
class LinearHeadModel(nn.Module):
    def __init__(self, input_dim, num_classes=2):
        super().__init__()
        self.linearhead = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linearhead(x)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def test_model(model, device, loader, criterion, set_name="test"):
    model.eval()
    total_loss, correct, total_samples = 0, 0, 0

    with torch.no_grad():
        for data_sample in loader:
            try:
                data, target = data_sample
            except:
                data, target, _, _ = data_sample
            data, target = data.to(device).float(), target.to(device).long()
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total_samples += data.size(0)

    avg_loss = total_loss / total_samples
    accuracy = 100. * correct / total_samples
    print(f'{set_name.capitalize()} — Loss: {avg_loss:.4f}, Acc: {accuracy:.2f}% ({correct}/{total_samples})')
    return accuracy


def erm_train(model, device, loader, optimizer, criterion, epoch):
    model.train()
    for batch_idx, data_sample in enumerate(tqdm(loader, desc=f"Epoch {epoch}")):
        try:
            data, target = data_sample
        except:
            data, target, _, _ = data_sample
        data, target = data.to(device).float(), target.to(device).long()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

def train_and_test_erm_animals(model_name, dataset_name, correlation, seed, embedding_save_dir,
                               checkpoint_root, id_classes, ood_classes, subset_key,
                               batch_size=64, num_epochs=30, num_classes=2):
    
    print(f"\nRunning ERM for model: {model_name}, correlation: {correlation}, seed: {seed}, OOD: {ood_classes}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Load data using your custom loader ===
    train_loader, val_loader, test_loader, _ = get_animal_loaders(
        model_name=model_name,
        batch_size=batch_size,
        ood_classes=ood_classes,
        low_shot=False,
        num_min_samples=None,
        seed=seed
    )

    # === Load everything into memory ===
    def loader_to_tensor(dataloader):
        all_embs, all_labels = [], []
        for emb, label, _ in dataloader:
            all_embs.append(emb)
            all_labels.append(label)
        return torch.cat(all_embs), torch.cat(all_labels)

    train_emb, train_labels = loader_to_tensor(train_loader)
    val_emb, val_labels = loader_to_tensor(val_loader)
    test_emb, test_labels = loader_to_tensor(test_loader)

    # === Map full class set to indices and filter for ID classes only ===
    all_classes = sorted(id_classes + list(ood_classes))
    class_to_idx = {cls: i for i, cls in enumerate(all_classes)}
    id_class_indices = [class_to_idx[cls] for cls in id_classes]

    def filter_id(emb, labels):
        mask = torch.isin(labels, torch.tensor(id_class_indices, device=labels.device))
        filtered_labels = labels[mask]
        remapped = torch.tensor([id_class_indices.index(l.item()) for l in filtered_labels], device=labels.device)
        return emb[mask], remapped

    train_emb, train_labels = filter_id(train_emb, train_labels)
    val_emb, val_labels = filter_id(val_emb, val_labels)
    test_emb, test_labels = filter_id(test_emb, test_labels)

    # === Create Dataloaders ===
    def create_dataloader(emb, labels, shuffle=False):
        dataset = torch.utils.data.TensorDataset(emb, labels)
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    train_loader = create_dataloader(train_emb, train_labels, shuffle=True)
    val_loader = create_dataloader(val_emb, val_labels)
    test_loader = create_dataloader(test_emb, test_labels)

    # === Define model ===
    input_dim = train_emb.shape[1]
    model = nn.Linear(input_dim, num_classes).to(device)
    optimizer = optim.SGD(model.parameters(), lr=1e-3, weight_decay=1e-3, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    save_dir = os.path.join(checkpoint_root, dataset_name, model_name)
    os.makedirs(save_dir, exist_ok=True)

    # === Training loop ===
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0
        for emb_batch, label_batch in train_loader:
            emb_batch, label_batch = emb_batch.to(device), label_batch.to(device)

            optimizer.zero_grad()
            output = model(emb_batch)
            loss = criterion(output, label_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * emb_batch.size(0)

        avg_loss = total_loss / len(train_loader.dataset)

        # === Validation ===
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for emb_batch, label_batch in val_loader:
                emb_batch, label_batch = emb_batch.to(device), label_batch.to(device)
                preds = model(emb_batch).argmax(dim=1)
                correct += (preds == label_batch).sum().item()
                total += label_batch.size(0)

        val_acc = correct / total
        print(f"Epoch {epoch:02d} | Train Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            ckpt_name = f'{model_name}_{dataset_name}_{correlation}_best_checkpoint_seed{seed}_{subset_key}.model'
            save_path = os.path.join(save_dir, ckpt_name)
            torch.save(model.state_dict(), save_path)
            print(f"> Saved best model @ {save_path}")

    # === Final test ===
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for emb_batch, label_batch in test_loader:
            emb_batch, label_batch = emb_batch.to(device), label_batch.to(device)
            preds = model(emb_batch).argmax(dim=1)
            correct += (preds == label_batch).sum().item()
            total += label_batch.size(0)

    test_acc = correct / total
    print(f"Best Val Acc: {best_acc:.4f} | Final Test Acc: {test_acc:.4f}")

    return test_acc

def train_and_test_erm(model_name, dataset_name, correlation, seed , embedding_save_dir, checkpoint_root, batch_size=64, num_epochs=30, num_classes=2):
    print(f"\nRunning ERM for model: {model_name}, correlation: {correlation}, seed: {seed}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_abrv = {
    'waterbirds': 'wb',
    'celeba_blond': 'celeba',
    'animals_metacoco': 'animals',
    'urbancars': 'urb',
    'spurious_imagenet': 'sp_imagenet'
    }
    
    if 'celeba' in dataset_name:
        emb_path = f'{embedding_save_dir}/{dataset_name}/{dataset_abrv[dataset_name]}_embs_{model_name}_{correlation}_seed{seed}.npy'
    else:
        emb_path = f'{embedding_save_dir}/{dataset_name}/{dataset_abrv[dataset_name]}_embs_{model_name}_{correlation}.npy'

    if dataset_name == 'urbancars':
        (train_emb, train_labels,train_place, train_co), (val_emb, val_labels, val_place, val_co), (test_emb, test_labels, test_place, test_co) = load_embeddings_and_labels_urb(emb_path)
        train_loader, val_loader, test_loader = create_dataloaders(
        (train_emb, train_labels,train_place, train_co), (val_emb, val_labels, val_place, val_co), (test_emb, test_labels, test_place, test_co), batch_size, is_urb=True)
    else:
        (train_emb, train_labels), (val_emb, val_labels), (test_emb, test_labels) = load_embeddings_and_labels(emb_path)
        train_loader, val_loader, test_loader = create_dataloaders(
        (train_emb, train_labels), (val_emb, val_labels), (test_emb, test_labels), batch_size)
    print(train_emb.shape)
    print("*"*100)
    model = LinearHeadModel(input_dim=train_emb.shape[1], num_classes=num_classes).to(device)
    optimizer = optim.SGD(model.parameters(), lr=1e-3, weight_decay=1e-3, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    save_dir = os.path.join(checkpoint_root, dataset_name , model_name)
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(1, num_epochs + 1):
        erm_train(model, device, train_loader, optimizer, criterion, epoch)
        val_acc = test_model(model, device, val_loader, criterion, set_name=f'Validation Epoch {epoch}')

        if val_acc > best_acc:
            best_acc = val_acc
            save_path = os.path.join(save_dir, f'{model_name}_{dataset_name}_{correlation}_best_checkpoint_seed{seed}.model')
            torch.save(model.state_dict(), save_path)
            print(f"> Saved best model @ {save_path}")

    test_model(model, device, test_loader, criterion, set_name="Test")

