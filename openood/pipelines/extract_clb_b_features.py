import os
import tqdm
import yaml
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, parent_dir)

from openood.datasets.celebA_blond_dataset import celebABlondDataset, get_transform_celeba
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

data_path = os.path.join(dataset_root,'CelebA_dataset')

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


seeds =  range(20,26)
correlations = [0.9, 0.5]
splits = [0, 1, 2] 
split_map = {0: 'train', 1: 'val', 2: 'test'}

transform = get_transform_celeba(False)

for seed in seeds:
    for correlation in correlations:
        for model_name, model in pretrained_models.items():
            # for finetuned
            # PATH = os.path.join(finetuned_checkpoint_root, f'celeba_blond/{model_name}/resnet50_celeba_blond_{correlation}_best_checkpoint_seed{seed}.model')
            # checkpoint = torch.load(PATH)
            # model.load_state_dict(checkpoint)
            ###
            model.to(device)
            model.eval()
            emb_dict = {}

            for split in splits:
                split_str = split_map[split]
                dataset = celebABlondDataset(split=split_str, seed=seed, path= data_path, data_label_correlation=correlation, transform=transform)
                dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)

                for i, (images, labels, genders) in enumerate(tqdm.tqdm(dataloader)):
                    images = images.to(device)
                    with torch.no_grad():
                        _, embeddings = model(images, return_feature=True)
                    embeddings = embeddings.cpu().numpy()

                    for j, emb in enumerate(embeddings):
                        name = f'{labels[j].item()}_{genders[j].item()}_{split}_{i * dataloader.batch_size + j}'
                        emb_dict[name] = emb

            # emb_save_path = os.path.join(finetuned_embeddings_dir, f'celeba_blond/celeba_embs_{model_name}_{correlation}_seed{seed}.npy')
            emb_save_path = os.path.join(embedding_save_dir, f'celeba_blond/celeba_embs_{model_name}_{correlation}_seed{seed}.npy')

            os.makedirs(os.path.dirname(emb_save_path), exist_ok=True)
            np.save(emb_save_path, emb_dict)
            print(f'Saved embeddings for {model_name} at seed {seed}, correlation {correlation}')
