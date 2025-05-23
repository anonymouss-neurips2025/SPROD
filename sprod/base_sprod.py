from typing import Any, Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

import openood.utils.comm as comm
from openood.postprocessors.base_postprocessor import BasePostprocessor


class BaseSPROD(BasePostprocessor):
    def __init__(self, config, probabilistic_score: bool = False, normalize_features: bool = True):
        super(BaseSPROD, self).__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        self.setup_flag = False

        self.probabilistic_score = probabilistic_score
        self.normalize_features = normalize_features

        self.train_labels = None
        self.train_feats = None
        self.train_prototypes = []

    def normalize(self, x: np.ndarray, eps: float = 1e-7) -> np.ndarray:
        """Normalize input vectors to unit norm."""
        return x / (np.linalg.norm(x, axis=-1, keepdims=True) + eps)

    def calc_euc_dist(self, embs: np.ndarray, prototypes: np.ndarray) -> np.ndarray:
        """Compute Euclidean distances between embeddings and prototypes."""
        return np.linalg.norm(embs[:, None, :] - prototypes[None, :, :], axis=-1)

    def _collect_features_and_labels(
        self,
        loader_source: Any,
        multi_loader: bool = False,
        progress: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Collect features and labels from one or more dataloaders."""
        features, labels = [], []

        loaders = (
            loader_source.items() if multi_loader
            else [(None, loader_source)]
        )

        for _, data_loader in loaders:
            for batch in tqdm(data_loader, disable=not progress or not comm.is_main_process()):
                data, label = batch[0], batch[1]
                features.append(data)
                labels.append(label)

        features = np.concatenate(features)
        labels = np.concatenate(labels).astype(int)

        if self.normalize_features:
            features = self.normalize(features)

        return features, labels


    def build_class_prototypes(self, features: np.ndarray, labels: np.ndarray):
        """Build class prototypes from features and labels."""
        self.train_prototypes = []
        for c in np.unique(labels):
            inds = np.where(labels == c)[0]
            class_embs = features[inds]
            prototype = class_embs.mean(axis=0)
            self.train_prototypes.append(prototype)
            

    def setup(self, net: nn.Module, id_loader_dict: dict, ood_loader_dict: dict, model_name):
        """Setup phase: extract features and build prototypes."""
        net.eval()
        with torch.no_grad():
            features, labels = self._collect_features_and_labels(id_loader_dict['train'])

        self.train_feats = features
        self.train_labels = labels
        self.build_class_prototypes(features, labels)

        self.setup_flag = True

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any, use_features: bool = True, model_name: str = ''):
        """Optional post-processing after setup. Can be overridden."""
        raise NotImplementedError("postprocess() must be overridden in a subclass if needed.")

    def inference(
        self,
        net: nn.Module,
        data_loaders_dict: dict,
        progress: bool = True,
        use_features: bool = False,
        model_name: str = ''
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run inference: predict classes and confidence scores."""
        net.eval()
        with torch.no_grad():
            data, labels = self._collect_features_and_labels(
                data_loaders_dict,
                multi_loader=True,
                progress=progress
            )

        prototypes = np.stack(self.train_prototypes)
        dists = self.calc_euc_dist(data, prototypes)

        if self.probabilistic_score:
            probs = np.exp(-dists)
            probs = probs / np.sum(probs, axis=1, keepdims=True)
            dists = 1.0 - probs

        pred_list = np.argmin(dists, axis=1).astype(int)
        conf_list = -np.min(dists, axis=1)

        return pred_list, conf_list, labels

    def numpy_inference(self,
                        embeddings: np.ndarray,
                        labels: np.ndarray = None,
                        donormalize: bool = False) :
    
        if donormalize:
            embeddings = self.normalize(embeddings)
    
        if labels is not None:
            self.build_class_prototypes(embeddings, labels)
            self.setup_flag = True
    
        prototypes = np.stack(self.train_prototypes)
        dists = self.calc_euc_dist(embeddings, prototypes)
    
        if self.probabilistic_score:
            probs = np.exp(-dists)
            probs = probs / np.sum(probs, axis=1, keepdims=True)
            dists = 1.0 - probs
    
        pred_list = np.argmin(dists, axis=1).astype(int)
        conf_list = -np.min(dists, axis=1)
    
        return pred_list, conf_list, labels
