import os
import urllib.request

from openood.postprocessors import (
    ASHPostprocessor, BasePostprocessor, ConfBranchPostprocessor,
    CutPastePostprocessor, DICEPostprocessor, DRAEMPostprocessor,
    DropoutPostProcessor, DSVDDPostprocessor, EBOPostprocessor,
    EnsemblePostprocessor, GMMPostprocessor, GodinPostprocessor,
    GradNormPostprocessor, GRAMPostprocessor, KLMatchingPostprocessor,
    KNNPostprocessor, MaxLogitPostprocessor, MCDPostprocessor,
    MDSPostprocessor, MDSEnsemblePostprocessor, MOSPostprocessor,
    ODINPostprocessor, OpenGanPostprocessor, OpenMax, PatchcorePostprocessor,
    Rd4adPostprocessor, ReactPostprocessor, ResidualPostprocessor,
    ScalePostprocessor, SSDPostprocessor, TemperatureScalingPostprocessor,
    VIMPostprocessor, RotPredPostprocessor, RankFeatPostprocessor,
    RMDSPostprocessor, SHEPostprocessor, CIDERPostprocessor, NPOSPostprocessor,
    GENPostprocessor, NNGuidePostprocessor, RelationPostprocessor,
    T2FNormPostprocessor, ReweightOODPostprocessor, fDBDPostprocessor,
    AdaScalePostprocessor, IODINPostprocessor, NCIPostprocessor)
from sprod.sprod1 import SPROD1
from sprod.sprod2 import SPROD2
from sprod.sprod3 import SPROD3
from openood.utils.config import Config, merge_configs

postprocessors = {
    'nci': NCIPostprocessor,
    'fdbd': fDBDPostprocessor,
    'ash': ASHPostprocessor,
    'cider': CIDERPostprocessor,
    'conf_branch': ConfBranchPostprocessor,
    'msp': BasePostprocessor,
    'ebo': EBOPostprocessor,
    'odin': ODINPostprocessor,
    'iodin': IODINPostprocessor,
    'mds': MDSPostprocessor,
    'mds_ensemble': MDSEnsemblePostprocessor,
    'npos': NPOSPostprocessor,
    'rmds': RMDSPostprocessor,
    'gmm': GMMPostprocessor,
    'patchcore': PatchcorePostprocessor,
    'openmax': OpenMax,
    'react': ReactPostprocessor,
    'vim': VIMPostprocessor,
    'gradnorm': GradNormPostprocessor,
    'godin': GodinPostprocessor,
    'mds': MDSPostprocessor,
    'gram': GRAMPostprocessor,
    'cutpaste': CutPastePostprocessor,
    'mls': MaxLogitPostprocessor,
    'residual': ResidualPostprocessor,
    'klm': KLMatchingPostprocessor,
    'temp_scaling': TemperatureScalingPostprocessor,
    'ensemble': EnsemblePostprocessor,
    'dropout': DropoutPostProcessor,
    'draem': DRAEMPostprocessor,
    'dsvdd': DSVDDPostprocessor,
    'mos': MOSPostprocessor,
    'mcd': MCDPostprocessor,
    'opengan': OpenGanPostprocessor,
    'knn': KNNPostprocessor,
    'dice': DICEPostprocessor,
    'scale': ScalePostprocessor,
    'ssd': SSDPostprocessor,
    'she': SHEPostprocessor,
    'rd4ad': Rd4adPostprocessor,
    'rotpred': RotPredPostprocessor,
    'rankfeat': RankFeatPostprocessor,
    'gen': GENPostprocessor,
    'nnguide': NNGuidePostprocessor,
    'relation': RelationPostprocessor,
    't2fnorm': T2FNormPostprocessor,
    'reweightood': ReweightOODPostprocessor,
    'adascale_a': AdaScalePostprocessor,
    'adascale_l': AdaScalePostprocessor,
    'sprod1': SPROD1,
    'sprod2': SPROD2,
    'sprod3': SPROD3
}

link_prefix = 'https://raw.githubusercontent.com/Jingkang50/OpenOOD/main/configs/postprocessors/'


def get_postprocessor(config_root: str, postprocessor_name: str,
                      id_data_name: str):
    postprocessor_config_path = os.path.join(config_root, 'postprocessors',
                                             f'{postprocessor_name}.yml')
    print(postprocessor_config_path)
    if not os.path.exists(postprocessor_config_path):
        os.makedirs(os.path.dirname(postprocessor_config_path), exist_ok=True)
        urllib.request.urlretrieve(link_prefix + f'{postprocessor_name}.yml',
                                   postprocessor_config_path)

    config = Config(postprocessor_config_path)

    config = merge_configs(config,
                           Config(**{'dataset': {
                               'name': id_data_name
                           }}))
    postprocessor = postprocessors[postprocessor_name](config)
    postprocessor.APS_mode = config.postprocessor.APS_mode
    postprocessor.hyperparam_search_done = False
    return postprocessor
