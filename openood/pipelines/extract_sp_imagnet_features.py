import os
import tqdm
import yaml
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, parent_dir)

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
dataset_root = os.path.join(paths["ID_data"], "OOD_Datasets/spurious_imagenet/dataset/spurious_imagenet")
embedding_save_dir = paths["ID_emb"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

splits = [0, 1, 2]
split_map = {0: 'train', 1: 'val', 2: 'test'}



for model_name, model in pretrained_models.items():
        model.to(device)
        model.eval()
        emb_dict = defaultdict(dict)

        for split in splits:
            dataloader = get_spurious_imagenet_loader(
                split=split_map[split],
                batch_size=32,
                root_dir=dataset_root,
                shuffle=False
            )

            for i, (images, labels) in enumerate(tqdm.tqdm(dataloader, desc=f"{model_name}-{split}")):
                images = images.to(device)
                with torch.no_grad():
                    _, features = model(images, return_feature=True)
                features = features.cpu().numpy()
                labels = labels.cpu().numpy()

                for j, emb in enumerate(features):
                    name = f'{labels[j]}_0_{split}_{i * dataloader.batch_size + j}'
                    emb_dict[name] = emb

        save_path = f"{embedding_save_dir}/spurious_imagenet/sp_imagenet_embs_{model_name}.npy"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, emb_dict)
        print(f"Saved features for {model_name}")
