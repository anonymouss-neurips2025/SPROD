import os
import sys
from pathlib import Path

import yaml
import tqdm
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, parent_dir)

from openood.datasets.cub_dataset import get_transform_cub
from openood.datasets.celebA_blond_dataset import celebAOodDataset
from openood.datasets.urbancars_dataset import JustBGOOD, NoCarOOD
from openood.datasets.svhn_loader import SVHN
from openood.datasets.spurious_imagenet_dataset import get_spurious_imagenet_loader

from openood.networks.models import (
    ResNet18, ResNet50, ResNet34, ResNet101,
    BiT_M_R50x1, BiT_M_R50x3, BiT_M_R101x1,
    DeiT_Ti, DeiT_S, DeiT_B,
    ViT_Ti, ViT_S, ViT_B,
    DINOv2_ViT_S_14, DINOv2_ViT_B_14, DINOv2_ViT_L_14,
    CLIP_ViT_B_16, CLIP_RN50,
    Swin_T, Swin_S, Swin_B,
    ConvNeXt_S, ConvNeXt_B, ConvNeXt_T
)

repo_root = Path(__file__).resolve().parents[2]
config_path = repo_root / "configs/sprod_paths.yaml"

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

paths = config["paths"]
dataset_root = Path(paths["ood_data"])
clb_root = Path(paths["ID_data"])
embedding_save_dir = Path(paths["ood_emb"])
urb_val_test_root = Path(paths["urbancars_val_test"])
sp_imgnet_root = os.path.join(paths["ID_data"], "OOD_Datasets/spurious_imagenet/dataset/spurious_imagenet")
finetuned_checkpoint_root = paths["finetuned_checkpoints"]
finetuned_embeddings_dir = paths['finetuned_embeddings']


pretrained_models = {
    'resnet_18': ResNet18(),
    'resnet_34': ResNet34(),
    'resnet_50': ResNet50(),
    'resnet_101': ResNet101(),
    'BiT_M_R50x1': BiT_M_R50x1(),
    'BiT_M_R50x3': BiT_M_R50x3(),
    'BiT_M_R101x1': BiT_M_R101x1(),
    'DeiT_Ti': DeiT_Ti(),
    'DeiT_S': DeiT_S(),
    'DeiT_B': DeiT_B(),
    'ViT_Ti': ViT_Ti(),
    'ViT_S': ViT_S(),
    'ViT_B': ViT_B(),
    'dinov2_vits14': DINOv2_ViT_S_14(),
    'dinov2_vitb14': DINOv2_ViT_B_14(),
    'dinov2_vitl14': DINOv2_ViT_L_14(),
    'clip_ViT_B_16': CLIP_ViT_B_16(),
    'clip_RN50': CLIP_RN50(),
    'Swin_T': Swin_T(),
    'Swin_S': Swin_S(),
    'Swin_B': Swin_B(),
    'ConvNeXt_T': ConvNeXt_T(),
    'ConvNeXt_S': ConvNeXt_S(),
    'ConvNeXt_B': ConvNeXt_B()


}

ood_datasets = {
    'SVHN': dataset_root / 'SVHN',
    'iSUN': dataset_root / 'iSUN',
    'LSUN_resize': dataset_root / 'LSUN_resize',
    'textures': dataset_root / 'textures',
    'clbood': clb_root / 'CelebA_dataset',
    'placesbg': dataset_root / 'placesbg',
    'urbn_just_bg_ood': urb_val_test_root, 
    'urbn_no_car_ood': urb_val_test_root,
    'spurious_imagenet': sp_imgnet_root
}

transform = get_transform_cub(train=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seeds = range(20, 26)
correlation = 0.9
ID_name = 'celeba_blond'
fine_tuned = False

for model_name, model in pretrained_models.items():
    # for seed in seeds:
            # for finetuned
            # PATH = os.path.join(finetuned_checkpoint_root, f'{ID_name}/{model_name}/{"".join(model_name.split("_")[:2])}_{ID_name}_{correlation}_best_checkpoint_seed{seed}.model')
            # checkpoint = torch.load(PATH)
            # model.load_state_dict(checkpoint)
            ####
            seed = 10
            model.to(device)
            model.eval()
            if fine_tuned:
                finetuned_embeddings_dir = Path(finetuned_embeddings_dir)
                emb_save_path = finetuned_embeddings_dir / f"ood_embeddings_{model_name}_{ID_name}_{correlation}_seed{seed}.npy"
            else:
                emb_save_path = embedding_save_dir / f"ood_embeddings_{model_name}.npy"

            if emb_save_path.exists():
                existing_embeddings = np.load(emb_save_path, allow_pickle=True).item()
            else:
                existing_embeddings = {}

            ood_embeddings = existing_embeddings  

            for dataset_name, dataset_path in ood_datasets.items():
                if dataset_name == 'clbood':
                    testsetout = celebAOodDataset(dataset_path)
                elif dataset_name == "SVHN":
                    testsetout = SVHN(str(dataset_path), split='test', transform=transform, download=False)
                elif dataset_name == 'urbn_just_bg_ood':
                    testsetout = JustBGOOD(transform=transform, root_dir=dataset_path)
                elif dataset_name == 'urbn_no_car_ood':
                    testsetout = NoCarOOD(transform=transform, root_dir=dataset_path)
                elif dataset_name == 'spurious_imagenet':
                    dataloader = get_spurious_imagenet_loader(
                    split='',
                    batch_size=32,
                    root_dir=str(dataset_path),
                    is_ood=True,
                    shuffle=False
                )
                else:
                    testsetout = ImageFolder(str(dataset_path), transform=transform)

                if dataset_name != 'spurious_imagenet':
                    dataloader = DataLoader(testsetout, batch_size=32, shuffle=False, num_workers=4)

                emb_dict = {}
                for i, batch  in enumerate(tqdm.tqdm(dataloader, desc=f"{model_name} - {dataset_name}")):
                    with torch.no_grad():
                        if isinstance(batch, (tuple, list)):
                            images = batch[0].to(device)
                        else:
                            images = batch.to(device)
                        _, embs = model(images, return_feature=True)
                        for j, emb in enumerate(embs):
                            img_id = i * 32 + j
                            emb_dict[img_id] = emb.cpu().numpy()

                ood_embeddings[dataset_name] = {**ood_embeddings.get(dataset_name, {}), **emb_dict}
                print(f"[INFO] Processed {dataset_name}")
            
            os.makedirs(os.path.dirname(emb_save_path), exist_ok=True)
            np.save(emb_save_path, ood_embeddings)
            print(f"[✓] Saved OOD embeddings for {model_name} → {emb_save_path}")
