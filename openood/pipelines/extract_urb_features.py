import os
import tqdm
import yaml
import torch
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, parent_dir)
from openood.datasets.urbancars_dataset import get_urbancars_dataloader
from openood.networks.models import (
    ResNet18, ResNet50,ResNet34, ResNet101,
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
train_data_root = paths['urbancars_train']
testval_data_root = paths['urbancars_val_test']
checkpoint_root = paths["checkpoints"]
embedding_save_dir = paths["ID_emb"]




pretrained_models = {
    'resnet_18': ResNet18(),
    'resnet_50': ResNet50(),
    'resnet_34': ResNet34(),
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

correlations = [95]
seeds = range(20,26)

splits = [0, 1, 2] 
split_map = {0: 'train', 1: 'val', 2: 'test'}
for correlation in correlations:
    for model_name, model in pretrained_models.items():

        # for seed in seeds:
            ## for finetuned
            # PATH = os.path.join(checkpoint_root, f'urbancars/{model_name}_pretrained_all_finetuned/{model_name}_urbancars_{correlation}_best_checkpoint_seed{seed}.model'
            # checkpoint = torch.load(PATH)
            # model.load_state_dict(checkpoint)
            #####
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            model.eval()

            emb_dict = {}

            for split in splits:
                dataloader  = get_urbancars_dataloader(split_map[split], 32, train_data_root, testval_data_root)
                dict = {'country': 0, 'urban':1}

                for i, (images, labels, place, co_occ) in enumerate(tqdm.tqdm(dataloader)):
                    images = images.to(device)
                    with torch.no_grad():
                        _, embeddings = model(images, return_feature=True)
                    embeddings = embeddings.cpu().numpy()
                    for j, emb in enumerate(embeddings):
                        name = f'{labels[j].item()}_{dict[place[j]]}_{dict[co_occ[j]]}_{split}_{i * dataloader.batch_size + j}'
                        emb_dict[name] = emb

            emb_save_path = os.path.join(embedding_save_dir,f'urbancars/urb_embs_{model_name}_{correlation}.npy')
            os.makedirs(os.path.dirname(emb_save_path), exist_ok=True)
            np.save(emb_save_path, emb_dict)
            print(f'Saved embeddings for {model_name} at correlation {correlation}')
