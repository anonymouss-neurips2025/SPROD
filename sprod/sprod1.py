from .base_sprod import BaseSPROD

class SPROD1(BaseSPROD):
    def __init__(self, config, probabilistic_score=False, normalize_features=True):
        super().__init__(config,
                         probabilistic_score=probabilistic_score,
                         normalize_features=normalize_features)
