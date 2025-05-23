import os
import yaml
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import random
import sys


import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, parent_dir)

from openood.evaluation_api import Evaluator
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
with open(repo_root / "configs/sprod_paths.yaml", "r") as f:
    config = yaml.safe_load(f)

paths = config["paths"]
checkpoint_root = paths["checkpoints"]
results_dir = Path(paths["ood_detection_results"])
pickle_dir = results_dir / "pickles"
pickle_dir.mkdir(parents=True, exist_ok=True)

def generate_combined_dataframe(list_of_dataframes):
    df_list = [df for (df, _) in list_of_dataframes]  
    mean_df = pd.concat(df_list).groupby(level=0).mean()
    std_df = pd.concat(df_list).groupby(level=0).std()
    return mean_df.round(2).astype(str) + ' ± ' + std_df.round(2).astype(str)

def load_model(model_name, model, dataset_name, correlation, seed,ood_classes):
    ood_str = ''.join(ood_classes)
    fc_ckpt = f"{checkpoint_root}/animals/{model_name}/{model_name}_animals_{correlation}_best_checkpoint_seed{seed}_{ood_str}.model"
    model.load_trained_fc(fc_ckpt)
    return model

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def evaluate_model(model, dataset_name, correlation, seed, pname, model_name, device, num_runs, grouped_ood_list):
    all_metrics = []

    for ood_classes in grouped_ood_list:
        model = load_model(model_name, model, dataset_name, correlation, seed, ood_classes)
        model = model.to(device)

        for run in range(num_runs):
            set_random_seed(seed)
            evaluator = Evaluator(
                model,
                id_name=dataset_name,
                preprocessor=None,
                postprocessor_name=pname,
                postprocessor=None,
                batch_size=4,
                seed=seed,
                correlation=correlation,
                model_name=model_name,
                use_features=True,
                ood_classes=ood_classes, 

            )
            metrics = evaluator.eval_ood(fsood=False)
            all_metrics.append((metrics, ood_classes))  # tuple of (df, ood_class)

    return all_metrics

def evaluate_and_save(model_name, model, dataset_name, correlation, seed, num_runs, device, pname, ood_class_list, n_leave):
    all_metrics = []
    grouped_ood_list = [ood_class_list[i:i + n_leave] for i in range(0, len(ood_class_list), n_leave)]

    results = evaluate_model(model, dataset_name, correlation, seed, pname, model_name, device, num_runs, grouped_ood_list)
    all_metrics.extend(results)

    save_detailed_pickles(all_metrics, dataset_name, model_name, pname, correlation, seed, pickle_dir)

    return all_metrics

def get_results(dataset_name, models, correlations, seeds, num_runs, device, pname, ood_class_list, n_leave):
    data_dict = {}
    for model_name, model in models.items():
        for correlation in correlations:
            key = f"{dataset_name}_{model_name}_{correlation}"
            print(f"\nProcessing {key}")
            all_metrics = []
            for seed in seeds:
                print(f"Seed {seed}")
                metrics = evaluate_and_save(model_name, model, dataset_name, correlation, seed, num_runs, device, pname, ood_class_list, n_leave)
                all_metrics.extend(metrics)

            combined_df = generate_combined_dataframe(all_metrics)
            data_dict[key] = combined_df
    return data_dict



def save_detailed_pickles(all_metrics, dataset_name, model_name, pname, correlation, seed, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    for metric_df, ood_classes in all_metrics: 
        for ood_testset in metric_df.index:
            row = metric_df.loc[ood_testset].to_dict()
            row = {k: round(v, 2) for k, v in row.items()}
            filename = f"{dataset_name}^{model_name}^{pname}^{ood_testset}^r{correlation}^s{seed}^{''.join(ood_classes)}^leave2^iter3.pkl"
            with open(output_dir / filename, "wb") as f:
                pickle.dump(row, f)

if __name__ == "__main__":
    dataset_name = 'animals_metacoco'
    correlations = [95]
    seeds = range(20,26)
    num_runs = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pnames = ['sprod3', 'mds', 'she', 'rmds', 'vim', 'gradnorm', 'sprod1', 'sprod2', 'msp', 'mls', 'klm', 'ebo', 'react']

    ood_class_list = [
    'cat', 'cow', 'dog', 'dolphin', 'elephant', 'fox', 'frog', 'giraffe',
    'goose', 'horse', 'lion', 'monkey', 'owl', 'rabbit', 'rat', 'seal', 'sheep',
    'squirrel', 'tiger', 'wolf', 'bear', 'kangaroo', 'lizard', 'ostrich', 'tortoise',
    'crab'
    ]

    ood_class_list = sorted(ood_class_list)
    n_leave = 2
    num_classes=24
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
        'clip_RN50': CLIP_RN50(num_classes=num_classes),
        'resnet_34': ResNet34(num_classes=num_classes),
        'Swin_T': Swin_T(num_classes=num_classes),
        'Swin_S': Swin_S(num_classes=num_classes),
        'Swin_B': Swin_B(num_classes=num_classes),
        'ConvNeXt_T': ConvNeXt_T(num_classes=num_classes),
        'ConvNeXt_S': ConvNeXt_S(num_classes=num_classes),
        'ConvNeXt_B': ConvNeXt_B(num_classes=num_classes)
    }

    for pname in pnames:
        print(f"\n{'*'*10} Evaluating Postprocessor: {pname} {'*'*10}")
        data_dict = get_results(dataset_name, pretrained_models, correlations, seeds, num_runs, device, pname, ood_class_list, n_leave)

        print("Saved pickle and Excel results for:", pname)
