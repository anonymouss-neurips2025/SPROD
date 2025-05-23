import os
import yaml
from pathlib import Path
import sys
import yaml


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

from linear_prob_utils import set_seed, train_and_test_erm

# --- Setup paths ---
repo_root = Path(__file__).resolve().parents[2]
with open(repo_root / "configs/sprod_paths.yaml", "r") as f:
    conf = yaml.safe_load(f)
    paths = conf["paths"]
    names = conf["names"]


checkpoint_root = paths["checkpoints"]
embedding_save_dir = paths["ID_emb"]
dataset_name = names['ID_name']
num_classes = int(names['num_classes'])

def main():
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
    correlations_dict ={
        'waterbirds': [50, 90],
        'celeba_blond': [0.9, 0.5],
        'urbancars': [95],
        'animals': [95],
        'spurious_imagenet': [95]
    }

    seeds = range(20,26)

    for correlation in correlations_dict[dataset_name]:
        for model_name in pretrained_models:
            for seed in seeds:
                set_seed(seed)
                train_and_test_erm(model_name, dataset_name, correlation, seed, embedding_save_dir, checkpoint_root, num_classes=num_classes)


if __name__ == "__main__":
    main()
