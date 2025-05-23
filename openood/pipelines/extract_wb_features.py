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
from openood.datasets.cub_dataset import get_transform_cub
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
dataset_root = paths['ID_data']
checkpoint_root = paths["checkpoints"]
finetuned_checkpoint_root = paths["finetuned_checkpoints"]
embedding_save_dir = paths["ID_emb"]
finetuned_embeddings_dir = paths['finetuned_embeddings']


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

correlations = [50,90]
seeds = range(20, 26)

for correlation in correlations:
    dataset_path = os.path.join(dataset_root, f'Waterbirds_dataset/waterbird_complete{correlation}_forest2water2')
    metadata_path = os.path.join(dataset_path, 'metadata.csv')
    metadata = pd.read_csv(metadata_path)
    
    img_ids = metadata['img_id'].tolist()
    file_names = metadata['img_filename'].tolist()
    labels = metadata['y'].tolist()
    splits = metadata['split'].tolist()
    places = metadata['place'].tolist()
    place_filenames = metadata['place_filename'].tolist()

    for model_name, model in pretrained_models.items():
        # for seed in seeds:
        #     # for finetuned
        #     PATH = os.path.join(finetuned_checkpoint_root, f'waterbirds/{model_name}/resnet50_waterbirds_{correlation}_best_checkpoint_seed{seed}.model')
        #     checkpoint = torch.load(PATH)
        #     model.load_state_dict(checkpoint)
            ###
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            model.eval()

            emb_dict = {}

            for i in tqdm.tqdm(range(len(file_names))):
                name = f'{labels[i]}_{places[i]}_{splits[i]}_{img_ids[i]}'
                image_path = os.path.join(dataset_path, file_names[i])
                with torch.no_grad():
                    image = Image.open(image_path).convert('RGB')
                    transform = get_transform_cub(False)
                    image_tensor = transform(image)
                    _, emb = model(image_tensor.unsqueeze(0).to(device), return_feature=True)
                    emb_dict[name] = emb.squeeze().cpu().numpy()

            emb_save_path = os.path.join(embedding_save_dir,f'waterbirds/wb_embs_{model_name}_{correlation}.npy')
            # emb_save_path = os.path.join(finetuned_embeddings_dir,f'waterbirds/wb_embs_{model_name}_{correlation}_seed{seed}.npy')

            # os.makedirs(embedding_save_dir, exist_ok=True)
            os.makedirs(os.path.dirname(emb_save_path), exist_ok=True)

            np.save(emb_save_path, emb_dict)
            print(f'Saved embeddings for {model_name} at correlation {correlation} ')
