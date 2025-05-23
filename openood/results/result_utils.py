import pickle
import random
import torch
import numpy as np
import pandas as pd
from openood.evaluation_api import Evaluator
import os



def load_model(model_name, model, dataset_name, correlation, seed, checkpoint_root):
    fc_ckpt = f'{checkpoint_root}/{dataset_name}/{model_name}/{model_name}_{dataset_name}_{correlation}_best_checkpoint_seed{seed}.model'
    model.load_trained_fc(fc_ckpt)
    return model


def generate_combined_dataframe(list_of_dataframes):
    mean_df = pd.concat(list_of_dataframes).groupby(level=0).mean()
    std_df = pd.concat(list_of_dataframes).groupby(level=0).std()
    return mean_df.round(2).astype(str) + ' ± ' + std_df.round(2).astype(str)


def get_resutls(dataset_name, models, correlations, seeds, num_runs, device, pname, batch_size, results_dir, checkpoint_root, flag, fine_tuned):
    data_dict = {}
    for model_name, model in models.items():
        for correlation in correlations:
            key = f"{dataset_name}_{model_name}_{correlation}"
            print(f"\nProcessing {key}")
            all_metrics = []
            for seed in seeds:
                    print(f"Seed {seed}")
                    metrics = evaluate_and_save(model_name, model, dataset_name, correlation, seed, num_runs, device, pname, batch_size, results_dir, checkpoint_root, flag, fine_tuned)
                    all_metrics.extend(metrics)



def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate_model(model, dataset_name, correlation, seed, pname, model_name, device, num_runs, batch_size, use_features = True, fine_tuned=False):
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
            batch_size=batch_size,
            seed=seed,
            correlation=correlation,
            model_name=model_name,
            use_features=use_features,
            fine_tuned=fine_tuned
        )
        metrics = evaluator.eval_ood(fsood=False)
        all_metrics.append(metrics)

    return all_metrics

def evaluate_and_save(model_name, model, dataset_name, correlation, seed, num_runs, device, pname, batch_size, results_dir, checkpoint_root, flag, fine_tuned):
    """
    Evaluate the model for a given seed, correlation, and dataset, and save results.
    """
    all_metrics = []
    if not fine_tuned:
        model = load_model(model_name, model, dataset_name, correlation, seed, checkpoint_root)
    else:
        mname = "".join(model_name.split("_")[:2])
        PATH = os.path.join(checkpoint_root, f'{dataset_name}/{model_name}/{mname}_{dataset_name}_{correlation}_best_checkpoint_seed{seed}.model')
        checkpoint = torch.load(PATH)
        model.load_state_dict(checkpoint)
    metrics = evaluate_model(model, dataset_name, correlation, seed, pname, model_name, device, num_runs, batch_size,fine_tuned=fine_tuned)
    all_metrics.extend(metrics)
    pickle_dir = results_dir / "pickles"

    save_detailed_pickles(all_metrics, dataset_name, model_name, pname, correlation, seed, pickle_dir, flag)
    return all_metrics


def save_detailed_pickles(all_metrics, dataset_name, model_name, pname, correlation, seed, output_dir, flag="default"):
    """
    Save each OOD test set result from a single run as a detailed pickle.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for metric_df in all_metrics: 
            for ood_testset in metric_df.index:
                row = metric_df.loc[ood_testset].to_dict()
                row = {k: round(v, 2) for k, v in row.items()}
                filename = f"{dataset_name}^{model_name}^{pname}^{ood_testset}^r{correlation}^s{seed}^{flag}.pkl"
                with open(output_dir / filename, "wb") as f:
                    pickle.dump(row, f)

