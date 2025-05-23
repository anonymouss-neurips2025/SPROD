import os
import tqdm
import yaml
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import sys
import pandas as pd
import torchvision.transforms as transforms

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, parent_dir)

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
embedding_save_dir = paths["ID_emb"]
animasl_data_root = paths['animals']

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



metadata_path = f'{dataset_root}/animals_metadata.csv'
metadata = pd.read_csv(metadata_path)

transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_image(img_path):
    img = Image.open(img_path).convert('RGB')
    img = transform(img)
    return img

def get_dataloader(metadata, batch_size=32):
    images = []
    ids = []

    for _, row in metadata.iterrows():
        print(row)
        img_path = row['address']
        img_id = row['Id']
        try:
            img = load_image(animasl_data_root+img_path)
            images.append(img)
            ids.append(img_id)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")

    images = torch.stack(images)
    dataset = torch.utils.data.TensorDataset(images, torch.tensor(ids))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for model_name, model in pretrained_models.items():
    model.to(device)
    model.eval()
    
    emb_dict = {}
    dataloader = get_dataloader(metadata, batch_size=1)

    for images, ids in tqdm.tqdm(dataloader):
        images = images.to(device)
        ids = ids.cpu().numpy()

        with torch.no_grad():
            _, embeddings = model(images, return_feature=True)
        embeddings = embeddings.cpu().numpy()

        for i, emb in enumerate(embeddings):
            img_id = ids[i]
            emb_dict[img_id] = emb

    emb_save_path = f'{embedding_save_dir}/animals/animals_embs_{model_name}_95.npy'
    os.makedirs(os.path.dirname(emb_save_path), exist_ok=True)
    np.save(emb_save_path, emb_dict)
    print(f'Saved embeddings for {model_name}')

