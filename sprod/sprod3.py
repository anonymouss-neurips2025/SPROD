from .full_sprod import FullSPROD

class SPROD3(FullSPROD):
    def __init__(self, config):
        super().__init__(config,
                         use_group_refinement=True,
                         merge_refined_prototypes=False,
                         refine_n_iter=1)
