import os
import yaml
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import random
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, parent_dir)

from openood.evaluation_api import Evaluator
from openood.networks.models import (
    ResNet18, ResNet50, ResNet101,
    BiT_M_R50x1, BiT_M_R50x3, BiT_M_R101x1,
    DeiT_Ti, DeiT_S, DeiT_B,
    ViT_Ti, ViT_S, ViT_B,
    DINOv2_ViT_S_14, DINOv2_ViT_B_14, DINOv2_ViT_L_14,
    CLIP_ViT_B_16
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
    mean_df = pd.concat(list_of_dataframes).groupby(level=0).mean()
    std_df = pd.concat(list_of_dataframes).groupby(level=0).std()
    return mean_df.round(2).astype(str) + ' ± ' + std_df.round(2).astype(str)

def load_model(model_name, model, dataset_name, correlation, seed):
    fc_ckpt = f'{checkpoint_root}/{dataset_name}/{model_name}/{model_name}_{dataset_name}_{correlation}_best_checkpoint_seed{seed}.model'
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

def evaluate_model(model, dataset_name, correlation, seed, pname, model_name, device, num_runs, few_shot, num_few_shot):
    all_metrics = []
    model = model.to(device)

    for run in range(num_runs):
        set_random_seed(seed)
        evaluator = Evaluator(
            model,
            id_name=dataset_name,
            preprocessor=None,
            postprocessor_name=pname,
            postprocessor=None,
            batch_size=32,
            seed=seed,
            correlation=correlation,
            model_name=model_name,
            use_features=True,
            low_shot=few_shot,
            num_min_samples=num_few_shot
        )
        metrics = evaluator.eval_ood(fsood=False)
        all_metrics.append(metrics)

    return all_metrics

def evaluate_and_save(model_name, model, dataset_name, correlation, seed, num_runs, device, pname, few_shot, num_few_shot):
    all_metrics = []
    model = load_model(model_name, model, dataset_name, correlation, seed)
    metrics = evaluate_model(model, dataset_name, correlation, seed, pname, model_name, device, num_runs, few_shot, num_few_shot)
    all_metrics.extend(metrics)

    save_detailed_pickles(all_metrics, dataset_name, model_name, pname, correlation, seed, pickle_dir, num_few_shot=num_few_shot)

    return all_metrics

def get_resutls(dataset_name, models, correlations, seeds, num_runs, device, pname, few_shot, num_few_shot_values):
    data_dict = {}
    for model_name, model in models.items():
        for correlation in correlations:
            for k in num_few_shot_values:
                key = f"{dataset_name}_{model_name}_{correlation}_fewshot{k}"
                print(f"Processing {key}")
                all_metrics = []
                for seed in seeds:
                    print(f"Seed {seed}")
                    metrics = evaluate_and_save(model_name, model, dataset_name, correlation, seed, num_runs, device, pname, few_shot, k)
                    all_metrics.extend(metrics)

                combined_df = generate_combined_dataframe(all_metrics)
                data_dict[key] = combined_df
    return data_dict


            
def save_detailed_pickles(all_metrics, dataset_name, model_name, pname, correlation, seed, output_dir, flag="default", num_few_shot=None):
    """
    Save detailed metrics to pickle files, including the number of few-shot examples in the flag.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for metric_df in all_metrics: 
        for ood_testset in metric_df.index:
            row = metric_df.loc[ood_testset].to_dict()
            row = {k: round(v, 2) for k, v in row.items()}
            
            # Add number of few-shot to the flag
            if num_few_shot is not None:
                flag_with_shot = f"{num_few_shot}shot"
            else:
                flag_with_shot = flag
            
            filename = f"{dataset_name}_{model_name}_{pname}_{ood_testset}_r{correlation}_s{seed}_{flag_with_shot}.pkl"
            
            with open(output_dir / filename, "wb") as f:
                pickle.dump(row, f)


if __name__ == "__main__":
    dataset_name = 'waterbirds'
    correlations = [50]
    seeds = range(10,30)
    num_runs = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pnames = ['sprod3','knn','mds']
    num_few_shot_values = list(range(1, 11))  

    pretrained_models = {
        'resnet_18': ResNet18(),
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
        'clip_ViT_B_16': CLIP_ViT_B_16()
    }

    for pname in pnames:
        print(f"\n{'*'*10} Evaluating Postprocessor: {pname} {'*'*10}")
        data_dict = get_resutls(dataset_name, pretrained_models, correlations, seeds, num_runs, device, pname, True, num_few_shot_values)
        print("Saved pickle and Excel results for:", pname)
