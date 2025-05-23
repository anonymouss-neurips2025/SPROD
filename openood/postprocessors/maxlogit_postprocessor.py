from typing import Any

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .base_postprocessor import BasePostprocessor


class MaxLogitPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super().__init__(config)
        self.args = self.config.postprocessor.postprocessor_args

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any, use_features, model_name):
        if use_features:
            if 'resnet' in model_name:
                output = net.model.fc(data)
            elif 'clip' in model_name:
                output = net.proj(data)
            else:
                output = net.model.head(data)
        else:
            output = net(data)
        conf, pred = torch.max(output, dim=1)
        return pred, conf
