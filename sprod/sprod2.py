from .full_sprod import FullSPROD

class SPROD2(FullSPROD):
    def __init__(self, config):
        super().__init__(config,
                         use_group_refinement=False,
                         merge_refined_prototypes=False,
                         refine_n_iter=0)
