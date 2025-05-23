import os
import yaml
import torch
from pathlib import Path
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, parent_dir)

import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')

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

from result_utils import get_resutls

repo_root = Path(__file__).resolve().parents[2]
with open(repo_root / "configs/sprod_paths.yaml", "r") as f:
    conf = yaml.safe_load(f)
    paths = conf["paths"]
    names = conf["names"]

dataset_name = names['ID_name']
flag = names['flag']
checkpoint_root = paths["checkpoints"]
finetuned_checkpoint_root = paths["finetuned_checkpoints"]
results_dir = Path(paths["ood_detection_results"])
num_classes = int(names['num_classes'])



if __name__ == "__main__":
    correlations_dict ={
        'waterbirds': [90],
        'celeba_blond': [0.9],
        'urbancars': [95],
        'animals': [95], #dummy
        'spurious_imagenet': [95] #dummy
        }
    batchsize_dict ={
        'waterbirds': 32,
        'celeba_blond': 128,
        'urbancars': 32,
        'animals': 16,
        'spurious_imagenet': 16
        }
    correlations = correlations_dict[dataset_name]
    seeds = range(20,26)
    num_runs = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pnames = ['sprod3','knn', 'sprod4', 'mds', 'she', 'rmds', 'vim', 'gradnorm', 'sprod1', 'sprod2', 'msp', 'mls', 'klm', 'ebo', 'react', 'openmax']

    fine_tuned = False
    pretrained_models = {
        'resnet_18': ResNet18(num_classes=num_classes),
        'resnet_34': ResNet34(num_classes=num_classes),
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
        'clip_RN50': CLIP_RN50(num_classes=num_classes),
        'Swin_T': Swin_T(num_classes=num_classes),
        'Swin_S': Swin_S(num_classes=num_classes),
        'Swin_B': Swin_B(num_classes=num_classes),
        'ConvNeXt_T': ConvNeXt_T(num_classes=num_classes),
        'ConvNeXt_S': ConvNeXt_S(num_classes=num_classes),
        'ConvNeXt_B': ConvNeXt_B(num_classes=num_classes)
    }

    for pname in pnames:
        print(f"\n{'*'*10} Evaluating Postprocessor: {pname} {'*'*10}")
        chkp_root = finetuned_checkpoint_root if fine_tuned else checkpoint_root
        flag = "finetuned" if fine_tuned else flag
        get_resutls(dataset_name, pretrained_models, correlations, seeds, num_runs, device, pname,
                                              batchsize_dict[dataset_name], results_dir, chkp_root, flag, fine_tuned)

        print("Saved pickle and csv results for:", pname)
