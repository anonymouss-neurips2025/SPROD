import torch
import torch.nn as nn
import numpy as np
import os
import random
import torch.optim as optim
from tqdm import tqdm
import argparse
from torchvision import transforms
import sys
from pathlib import Path
import yaml

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, parent_dir)

import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')

from openood.datasets.celebA_blond_dataset import get_celebA_blond_loaders
from openood.networks.models import (
    ResNet18, ResNet50, ResNet101,
    BiT_M_R50x1, BiT_M_R50x3, BiT_M_R101x1,
    DeiT_Ti, DeiT_S, DeiT_B,
    ViT_Ti, ViT_S, ViT_B,
    DINOv2_ViT_S_14, DINOv2_ViT_B_14, DINOv2_ViT_L_14,
    CLIP_ViT_B_16, CLIP_RN50
)

repo_root = Path(__file__).resolve().parents[2]
with open(repo_root / "configs/sprod_paths.yaml", "r") as f:
    conf = yaml.safe_load(f)
    paths = conf["paths"]
    names = conf["names"]

dataset_name = names['ID_name']
flag = names['flag']
checkpoint_root = paths["finetuned_checkpoints"]
num_classes = int(names['num_classes'])
dataset_root = paths['ID_data']

pretrained_models = {
        'resnet_18': ResNet18(num_classes=num_classes),
        'resnet_50': ResNet50(num_classes=num_classes),
        'resnet_101': ResNet101(num_classes=num_classes),
        'BiT_M_R50x1': BiT_M_R50x1(num_classes=num_classes),
        'BiT_M_R50x3': BiT_M_R50x3(num_classes=num_classes),
        'BiT_M_R101x1': BiT_M_R101x1(num_classes=num_classes),
        'DeiT_Ti': DeiT_Ti(num_classes=num_classes),
        'DeiT_S': DeiT_S(num_classes=num_classes),
        'DeiT_B': DeiT_B(num_classes=num_classes),
        'ViT_Ti': ViT_Ti(num_classes=num_classes),
        'ViT_S': ViT_S(num_classes=num_classes),
        'ViT_B': ViT_B(num_classes=num_classes),
        'dinov2_vits14': DINOv2_ViT_S_14(num_classes=num_classes),
        'dinov2_vitb14': DINOv2_ViT_B_14(num_classes=num_classes),
        'dinov2_vitl14': DINOv2_ViT_L_14(num_classes=num_classes),
        'clip_ViT_B_16': CLIP_ViT_B_16(num_classes=num_classes),
        'clip_RN50': CLIP_RN50(num_classes=num_classes)

    }



def test_model(model, device, test_loader, set_name="test set"):
    criterion = nn.CrossEntropyLoss(reduction='sum')
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target, env in test_loader:
            data, target = data.to(device).float(), target.to(device).long()
            output = model(data)

            test_loss += criterion(output, target).item()  
            pred = torch.argmax(output, dim=1)
            label = target
            correct += pred.eq(label.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        f'\nPerformance on {set_name}: Average loss: {test_loss}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset)})\n')
    return 100. * correct / len(test_loader.dataset)


def erm_train(model, device, train_loader, optimizer, epoch):
    criterion = nn.CrossEntropyLoss()
    model.train()
    for batch_idx, (data, target, env) in enumerate(tqdm(train_loader)):
        data, target = data.to(device).float(), target.to(device).long()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(
                f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader)}%)]\tLoss: {loss.item()}')


def train_and_test_erm(seed, args):
    print("ERM...\n")
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    all_train_loader, val_loader, test_loader = get_celebA_blond_loaders(
                                                           batch_size=args.batch_size, correlation=args.r, seed=seed, path=args.data_path,)

    model = pretrained_models['ViT_S'].to(device)
    optimizer = optim.SGD(model.parameters(), lr=1e-3, weight_decay=1e-3, momentum=0.9)

    train_acc = []
    val_acc = []
    test_acc = []
    best_acc = 0
    save_dir = os.path.join(checkpoint_root, dataset_name , 'ViT_S')
    os.makedirs(save_dir, exist_ok=True)
    for epoch in range(1, 31):
        erm_train(model, device, all_train_loader, optimizer, epoch)
        train_acc.append(test_model(model, device, all_train_loader, set_name=f'train set epoch {epoch}'))
        val_acc.append(test_model(model, device, val_loader, set_name=f'validation set epoch {epoch}'))
        if val_acc[-1] > best_acc:
            best_acc = val_acc[-1]
            torch.save(model.state_dict(), os.path.join(save_dir, 
                                                        'ViT_S_ celeba_blond_'+ str(args.r)+'_best_checkpoint_seed' + str(
                                                            args.seed) +  '.model'))





def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    default_data_path = os.path.join(dataset_root, 'CelebA_dataset')
    parser.add_argument("--data_path", type=str, default=default_data_path, help="data path")
    parser.add_argument("--dataset", type=str, default='celeba_blond')
    parser.add_argument("--batch_size", type=int, default=128, help="batch_size")
    parser.add_argument("--backbone_size", type=int, default=64)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--sampling_mode", type=str, default='top-k')
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--r", type=int, default=0.9)

    base_args = parser.parse_args()

    for seed in range(20, 26):
        args = argparse.Namespace(**vars(base_args))  
        args.seed = seed
        print(f"\n=== Running with seed {seed} ===")
        set_seed(seed)
        train_and_test_erm(seed, args)