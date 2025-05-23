from .base_sprod import BaseSPROD
import numpy as np

class FullSPROD(BaseSPROD):
    def __init__(self, config, 
                 use_group_refinement: bool = False,
                 merge_refined_prototypes: bool = False,
                 refine_n_iter: int = 1):
        super().__init__(config, probabilistic_score=False, normalize_features=True)
        self.use_group_refinement = use_group_refinement
        self.merge_refined_prototypes = merge_refined_prototypes
        self.refine_n_iter = refine_n_iter

    def setup(self, net, id_loader_dict, ood_loader_dict, model_name):
        """Override setup to perform flexible classification-aware prototype refinement."""
        super().setup(net, id_loader_dict, ood_loader_dict, model_name)
        self.perform_classification()

    def perform_classification(self):
        """Expand, optionally refine, and optionally merge prototypes."""
        train_prototypes = np.stack(self.train_prototypes)
        x_train = np.array(self.train_feats)
        y_train = np.array(self.train_labels)

        dists = self.calc_euc_dist(x_train, train_prototypes)
        y_hat_train = np.argmin(dists, axis=1)

        total_misc_inds = np.where(y_hat_train != y_train)[0]
        total_crr_inds = np.where(y_hat_train == y_train)[0]

        final_prototypes = []

        num_classes = len(np.unique(y_train))

        for l in range(num_classes):
            group_embs = []
            class_inds = np.where(y_train == l)[0]

            # Correctly classified embeddings
            class_crr_inds = np.intersect1d(class_inds, total_crr_inds, assume_unique=True)
            if len(class_crr_inds) > 0:
                group_embs.append(x_train[class_crr_inds])

            for j in range(num_classes):
                if j == l:
                    continue
                # Misclassified to class j
                class_misc_inds = np.intersect1d(class_inds, total_misc_inds, assume_unique=True)
                trg_lbl_inds = np.where(y_hat_train == j)[0]
                class_misc_inds_trg = np.intersect1d(class_misc_inds, trg_lbl_inds, assume_unique=True)

                if len(class_misc_inds_trg) > 0:
                    group_embs.append(x_train[class_misc_inds_trg])

            # Decide what to do with the group
            if self.use_group_refinement and group_embs:
                refined_prototypes = self.refine_group_prototypes(group_embs, n_iter=self.refine_n_iter)
            else:
                refined_prototypes = [embs.mean(axis=0) for embs in group_embs]

            if self.merge_refined_prototypes:
                merged_prototype = np.mean(refined_prototypes, axis=0)
                final_prototypes.append(merged_prototype)
            else:
                final_prototypes.extend(refined_prototypes)

        self.train_prototypes = final_prototypes

    def refine_group_prototypes(self, group_embs, n_iter: int = 1):
        """Refine prototypes iteratively."""
        all_embs = np.concatenate(group_embs)
        prototypes = [embs.mean(axis=0) for embs in group_embs]
        prototypes = np.array(prototypes)

        for _ in range(n_iter):
            dists = self.calc_euc_dist(all_embs, prototypes)
            labels = np.argmin(dists, axis=1)
            new_embs = [all_embs[labels == l] for l in np.unique(labels)]
            prototypes = [embs.mean(axis=0) for embs in new_embs]
            prototypes = np.array(prototypes)

        return prototypes
