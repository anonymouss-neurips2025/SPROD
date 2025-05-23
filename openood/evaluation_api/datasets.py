import os
import gdown
import zipfile
from pathlib import Path
import yaml
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
import torchvision as tvs
if tvs.__version__ >= '0.13':
    tvs_new = True
else:
    tvs_new = False

from openood.datasets.feature_dataset import FeatDataset, load_embeddings, load_embeddings_and_labels, load_embeddings_and_labels_urb, create_dataloaders
from openood.datasets.cub_dataset import  get_waterbird_loaders
from openood.datasets.animals_metacoco_dataset import get_animal_loaders
from openood.datasets.spurious_imagenet_dataset import get_spurious_imagenet_loader

from openood.datasets.svhn_loader import SVHN 

repo_root = Path(__file__).resolve().parents[2]
config_path = repo_root / "configs/sprod_paths.yaml"


with open(config_path, "r") as f:
    config = yaml.safe_load(f)

paths = config["paths"]
finetuned_embeddings_paths = paths['finetuned_embeddings']


# As used in the SPROD paper: in this code, near-OOD datasets represent spurious OOD,
# while far-OOD datasets are used as non-spurious OOD.

DATA_INFO = {
    'cifar10': {
        'num_classes': 10,
        'id': {
            'train': {
                'data_dir': 'images_classic/',
                'imglist_path': 'benchmark_imglist/cifar10/train_cifar10.txt'
            },
            'val': {
                'data_dir': 'images_classic/',
                'imglist_path': 'benchmark_imglist/cifar10/val_cifar10.txt'
            },
            'test': {
                'data_dir': 'images_classic/',
                'imglist_path': 'benchmark_imglist/cifar10/test_cifar10.txt'
            }
        },
        'csid': {
            'datasets': ['cifar10c'],
            'cinic10': {
                'data_dir': 'images_classic/',
                'imglist_path': 'benchmark_imglist/cifar10/val_cinic10.txt'
            },
            'cifar10c': {
                'data_dir': 'images_classic/',
                'imglist_path': 'benchmark_imglist/cifar10/test_cifar10c.txt'
            }
        },
        'ood': {
            'val': {
                'data_dir': 'images_classic/',
                'imglist_path': 'benchmark_imglist/cifar10/val_tin.txt'
            },
            'near': {
                'datasets': ['cifar100', 'tin'],
                'cifar100': {
                    'data_dir': 'images_classic/',
                    'imglist_path':
                    'benchmark_imglist/cifar10/test_cifar100.txt'
                },
                'tin': {
                    'data_dir': 'images_classic/',
                    'imglist_path': 'benchmark_imglist/cifar10/test_tin.txt'
                }
            },
            'far': {
                'datasets': ['mnist', 'svhn', 'texture', 'places365'],
                'mnist': {
                    'data_dir': 'images_classic/',
                    'imglist_path': 'benchmark_imglist/cifar10/test_mnist.txt'
                },
                'svhn': {
                    'data_dir': 'images_classic/',
                    'imglist_path': 'benchmark_imglist/cifar10/test_svhn.txt'
                },
                'texture': {
                    'data_dir': 'images_classic/',
                    'imglist_path':
                    'benchmark_imglist/cifar10/test_texture.txt'
                },
                'places365': {
                    'data_dir': 'images_classic/',
                    'imglist_path':
                    'benchmark_imglist/cifar10/test_places365.txt'
                },
            }
        }
    },
    'cifar100': {
        'num_classes': 100,
        'id': {
            'train': {
                'data_dir': 'images_classic/',
                'imglist_path': 'benchmark_imglist/cifar100/train_cifar100.txt'
            },
            'val': {
                'data_dir': 'images_classic/',
                'imglist_path': 'benchmark_imglist/cifar100/val_cifar100.txt'
            },
            'test': {
                'data_dir': 'images_classic/',
                'imglist_path': 'benchmark_imglist/cifar100/test_cifar100.txt'
            }
        },
        'csid': {
            'datasets': [],
        },
        'ood': {
            'val': {
                'data_dir': 'images_classic/',
                'imglist_path': 'benchmark_imglist/cifar100/val_tin.txt'
            },
            'near': {
                'datasets': ['cifar10', 'tin'],
                'cifar10': {
                    'data_dir': 'images_classic/',
                    'imglist_path':
                    'benchmark_imglist/cifar100/test_cifar10.txt'
                },
                'tin': {
                    'data_dir': 'images_classic/',
                    'imglist_path': 'benchmark_imglist/cifar100/test_tin.txt'
                }
            },
            'far': {
                'datasets': ['mnist', 'svhn', 'texture', 'places365'],
                'mnist': {
                    'data_dir': 'images_classic/',
                    'imglist_path': 'benchmark_imglist/cifar100/test_mnist.txt'
                },
                'svhn': {
                    'data_dir': 'images_classic/',
                    'imglist_path': 'benchmark_imglist/cifar100/test_svhn.txt'
                },
                'texture': {
                    'data_dir': 'images_classic/',
                    'imglist_path':
                    'benchmark_imglist/cifar100/test_texture.txt'
                },
                'places365': {
                    'data_dir': 'images_classic/',
                    'imglist_path':
                    'benchmark_imglist/cifar100/test_places365.txt'
                }
            },
        }
    },
    'imagenet200': {
        'num_classes': 200,
        'id': {
            'train': {
                'data_dir':
                'images_largescale/',
                'imglist_path':
                'benchmark_imglist/imagenet200/train_imagenet200.txt'
            },
            'val': {
                'data_dir': 'images_largescale/',
                'imglist_path':
                'benchmark_imglist/imagenet200/val_imagenet200.txt'
            },
            'test': {
                'data_dir':
                'images_largescale/',
                'imglist_path':
                'benchmark_imglist/imagenet200/test_imagenet200.txt'
            }
        },
        'csid': {
            'datasets': ['imagenet_v2', 'imagenet_c', 'imagenet_r'],
            'imagenet_v2': {
                'data_dir':
                'images_largescale/',
                'imglist_path':
                'benchmark_imglist/imagenet200/test_imagenet200_v2.txt'
            },
            'imagenet_c': {
                'data_dir':
                'images_largescale/',
                'imglist_path':
                'benchmark_imglist/imagenet200/test_imagenet200_c.txt'
            },
            'imagenet_r': {
                'data_dir':
                'images_largescale/',
                'imglist_path':
                'benchmark_imglist/imagenet200/test_imagenet200_r.txt'
            },
        },
        'ood': {
            'val': {
                'data_dir': 'images_largescale/',
                'imglist_path':
                'benchmark_imglist/imagenet200/val_openimage_o.txt'
            },
            'near': {
                'datasets': ['ssb_hard', 'ninco'],
                'ssb_hard': {
                    'data_dir':
                    'images_largescale/',
                    'imglist_path':
                    'benchmark_imglist/imagenet200/test_ssb_hard.txt'
                },
                'ninco': {
                    'data_dir': 'images_largescale/',
                    'imglist_path':
                    'benchmark_imglist/imagenet200/test_ninco.txt'
                }
            },
            'far': {
                'datasets': ['inaturalist', 'textures', 'openimage_o'],
                'inaturalist': {
                    'data_dir':
                    'images_largescale/',
                    'imglist_path':
                    'benchmark_imglist/imagenet200/test_inaturalist.txt'
                },
                'textures': {
                    'data_dir':
                    'images_classic/',
                    'imglist_path':
                    'benchmark_imglist/imagenet200/test_textures.txt'
                },
                'openimage_o': {
                    'data_dir':
                    'images_largescale/',
                    'imglist_path':
                    'benchmark_imglist/imagenet200/test_openimage_o.txt'
                },
            },
        }
    },

# ===== Large Scale Dataset =====    
    'imagenet': {
        'num_classes': 1000,
        'id': {
            'data_dir': 'Datasets/Imagenet1k_dataset/',
            'emb_path': 'Embeddings/Imagenet1k/',

            'train': {
                'data_dir': 'images_largescale/',
                'imglist_path': 'benchmark_imglist/imagenet/train_imagenet.txt'
            },
            'val': {
                'data_dir': 'images_largescale/',
                'imglist_path': 'benchmark_imglist/imagenet/val_imagenet.txt'
            },
            'test': {
                'data_dir': 'images_largescale/',
                'imglist_path': 'benchmark_imglist/imagenet/test_imagenet.txt'
            }
        },
        'csid': {
            'datasets':
            ['imagenet_v2', 'imagenet_c', 'imagenet_r', 'imagenet_es'],
            'imagenet_v2': {
                'data_dir': 'images_largescale/',
                'imglist_path':
                'benchmark_imglist/imagenet/test_imagenet_v2.txt'
            },
            'imagenet_c': {
                'data_dir': 'images_largescale/',
                'imglist_path':
                'benchmark_imglist/imagenet/test_imagenet_c.txt'
            },
            'imagenet_r': {
                'data_dir': 'images_largescale/',
                'imglist_path':
                'benchmark_imglist/imagenet/test_imagenet_r.txt'
            },
            'imagenet_es': {
                'data_dir': 'images_largescale/',
                'imglist_path':
                'benchmark_imglist/imagenet/test_imagenet_es.txt'
            },
        },
        'ood': {
            'data_dir': 'Datasets/OOD_Datasets/', 
            'emb_path': 'Embeddings/OOD/',

            'val': {
                'data_dir': 'images_largescale/',
                'imglist_path':
                'benchmark_imglist/imagenet/val_openimage_o.txt'
            },
            'near': {
                'datasets': ['spurious-imagenet'],

            },
            'far': {
                'datasets': ['inaturalist', 'textures', 'openimage_o', 'ninco'],
                'inaturalist': {
                    'data_dir':
                    'images_largescale/',
                    'imglist_path':
                    'benchmark_imglist/imagenet/test_inaturalist.txt'
                },
                'textures': {
                    'data_dir': 'images_classic/',
                    'imglist_path':
                    'benchmark_imglist/imagenet/test_textures.txt'
                },
                'openimage_o': {
                    'data_dir':
                    'images_largescale/',
                    'imglist_path':
                    'benchmark_imglist/imagenet/test_openimage_o.txt'
                },

                'ninco': {
                    'data_dir': 'images_largescale/',
                    'imglist_path': 'benchmark_imglist/imagenet/test_ninco.txt'
                }
            },
        }
    },


# ===================== Spurious Feature Datasets =====================
# The following datasets are added specifically for evaluating the
# SPROD method under various types of spurious correlations in OOD settings.
# =====================================================================



# ===== Synthetic Datasets =====

    'waterbirds': {
        'num_classes': 2,
        'id': {
            'data_dir': f"{paths['ID_data']}/Waterbirds_dataset/",
            'emb_path': f"{paths['ID_emb']}/waterbirds/"
        },

        'ood': {
            'data_dir': paths['ood_data'], 
            'emb_path': paths['ood_emb'],

            'near': {
                'datasets': ['placesbg']
            },
            'far': {
                'datasets': ['SVHN', 'iSUN', 'LSUN_resize']
            },
        }
    },

    'urbancars': {
        'num_classes': 2,
        'id': {
            'data_dir': f"{paths['ID_data']}/UrbanCars_dataset/",
            'emb_path': f"{paths['ID_emb']}/urbancars"
        },

        'ood': {
            'data_dir': paths['ood_data'], 
            'emb_path': paths['ood_emb'],

            'near': {
                'datasets': ['urbn_just_bg_ood', 'urbn_no_car_ood']
            },
            'far': {
                'datasets': ['SVHN', 'iSUN', 'LSUN_resize']
            },
        }
    },

# ===== Real-World Datasets =====

    'celeba_blond': {
        'num_classes': 2,
        'id': {
            'data_dir': f"{paths['ID_data']}/CelebA_dataset/",
            'emb_path': f"{paths['ID_emb']}/celeba_blond/"
        },

        'ood': {
            'data_dir': paths['ood_data'], 
            'emb_path': paths['ood_emb'],

            'near': {
                'datasets': ['clbood']
            },
            'far': {
                'datasets': ['SVHN', 'iSUN', 'LSUN_resize']
            },
        }
    },

    'animals_metacoco': {
        'num_classes': 24,
        'id': {
            'data_dir': 'Datasets/AnimalsMetaCoco/',
            'emb_path': 'Embeddings/AnimalsMetaCoco/'
        },

        'ood': {
            'data_dir': paths['ood_data'], 
            'emb_path': paths['ood_emb'],

            'near': {
                'datasets': ['animals_ood']
            },
            'far': {
                'datasets': ['SVHN', 'iSUN', 'LSUN_resize']
            },
        }
    },


    'spurious_imagenet': {
        'num_classes': 100,
        'id': {
            'data_dir': f"{paths['ood_data']}/spurious_imagenet",
            'emb_path': f"{paths['ID_emb']}/spurious_imagenet/"
        },

        'ood': {
            'data_dir': paths['ood_data'], 
            'emb_path': paths['ood_emb'],

            'near': {
                'datasets': ['spurious_imagenet']
            },
            'far': {
                'datasets': ['SVHN', 'iSUN', 'LSUN_resize']
            },
        }
    },
   

}

download_id_dict = {
    'cifar10': '1Co32RiiWe16lTaiOU6JMMnyUYS41IlO1',
    'cifar100': '1PGKheHUsf29leJPPGuXqzLBMwl8qMF8_',
    'tin': '1PZ-ixyx52U989IKsMA2OT-24fToTrelC',
    'mnist': '1CCHAGWqA1KJTFFswuF9cbhmB-j98Y1Sb',
    'svhn': '1DQfc11HOtB1nEwqS4pWUFp8vtQ3DczvI',
    'texture': '1OSz1m3hHfVWbRdmMwKbUzoU8Hg9UKcam',
    'places365': '1Ec-LRSTf6u5vEctKX9vRp9OA6tqnJ0Ay',
    'imagenet_1k': '1i1ipLDFARR-JZ9argXd2-0a6DXwVhXEj',
    'species_sub': '1-JCxDx__iFMExkYRMylnGJYTPvyuX6aq',
    'ssb_hard': '1PzkA-WGG8Z18h0ooL_pDdz9cO-DCIouE',
    'ninco': '1Z82cmvIB0eghTehxOGP5VTdLt7OD3nk6',
    'inaturalist': '1zfLfMvoUD0CUlKNnkk7LgxZZBnTBipdj',
    'places': '1fZ8TbPC4JGqUCm-VtvrmkYxqRNp2PoB3',
    'sun': '1ISK0STxWzWmg-_uUr4RQ8GSLFW7TZiKp',
    'openimage_o': '1VUFXnB_z70uHfdgJG2E_pjYOcEgqM7tE',
    'imagenet_v2': '1akg2IiE22HcbvTBpwXQoD7tgfPCdkoho',
    'imagenet_r': '1EzjMN2gq-bVV7lg-MEAdeuBuz-7jbGYU',
    'imagenet_c': '1JeXL9YH4BO8gCJ631c5BHbaSsl-lekHt',
    'imagenet_es': '1ATz11vKmPqyzfEaEDRaPTF9TXiC244sw',
    'benchmark_imglist': '1lI1j0_fDDvjIt9JlWAw09X8ks-yrR_H1'
}

dir_dict = {
    'images_classic/': [
        'cifar100', 'tin', 'tin597', 'svhn', 'cinic10', 'imagenet10', 'mnist',
        'fashionmnist', 'cifar10', 'cifar100c', 'places365', 'cifar10c',
        'fractals_and_fvis', 'usps', 'texture', 'notmnist'
    ],
    'images_largescale/': [
        'imagenet_1k',
        'ssb_hard',
        'ninco',
        'inaturalist',
        'places',
        'sun',
        'openimage_o',
        'imagenet_v2',
        'imagenet_c',
        'imagenet_r',
        'imagenet_es',
    ],
    'images_medical/': ['actmed', 'bimcv', 'ct', 'hannover', 'xraybone'],
}

benchmarks_dict = {
    'cifar10':
    ['cifar10', 'cifar100', 'tin', 'mnist', 'svhn', 'texture', 'places365'],
    'cifar100':
    ['cifar100', 'cifar10', 'tin', 'mnist', 'svhn', 'texture', 'places365'],
    'imagenet200': [
        'imagenet_1k', 'ssb_hard', 'ninco', 'inaturalist', 'texture',
        'openimage_o', 'imagenet_v2', 'imagenet_c', 'imagenet_r'
    ],
    'imagenet': [
        'imagenet_1k', 'ssb_hard', 'ninco', 'inaturalist', 'texture',
        'openimage_o', 'imagenet_v2', 'imagenet_c', 'imagenet_r', 'imagenet_es'
    ],
}


def require_download(filename, path):
    for item in os.listdir(path):
        if item.startswith(filename) or filename.startswith(
                item) or path.endswith(filename):
            return False

    else:
        print(filename + ' needs download:')
        return True


def download_dataset(dataset, data_root):
    for key in dir_dict.keys():
        if dataset in dir_dict[key]:
            store_path = os.path.join(data_root, key, dataset)
            if not os.path.exists(store_path):
                os.makedirs(store_path)
            break
    else:
        print('Invalid dataset detected {}'.format(dataset))
        return

    if require_download(dataset, store_path):
        print(store_path)
        if not store_path.endswith('/'):
            store_path = store_path + '/'
        gdown.download(id=download_id_dict[dataset], output=store_path)

        file_path = os.path.join(store_path, dataset + '.zip')
        with zipfile.ZipFile(file_path, 'r') as zip_file:
            zip_file.extractall(store_path)
        os.remove(file_path)


def data_setup(data_root, id_data_name):
    if not data_root.endswith('/'):
        data_root = data_root + '/'

    if not os.path.exists(os.path.join(data_root, 'benchmark_imglist')):
        gdown.download(id=download_id_dict['benchmark_imglist'],
                       output=data_root)
        file_path = os.path.join(data_root, 'benchmark_imglist.zip')
        with zipfile.ZipFile(file_path, 'r') as zip_file:
            zip_file.extractall(data_root)
        os.remove(file_path)

    for dataset in benchmarks_dict[id_data_name]:
        download_dataset(dataset, data_root)
import numpy as np


def get_id_ood_dataloader(id_name, model_name, batch_size, seed,  correlation, use_features, ood_val_source = "nsp", ood_classes = None, low_shot = False, num_min_samples = 1 , fine_tuned=False):
    """
    Parameters:
    ----------
    use_features : bool
        If True, loads precomputed image features/embeddings instead of raw image data.
        If False, loads and preprocesses raw images from disk.

    ood_val_source : str, optional (default="nsp")
        Specifies the source of OOD validation data. Options are:
            - "sp"       : Use spurious dataset for OOD validation.
            - "nsp"      : Use non-spurious (e.g., standard OOD datasets like SVHN, iSUN) for validation.
            - "both"     : Use both spurious and non-spurious sources for OOD validation.

    model_name : str
        Specifies the model used to generate the precomputed features. Required when `use_features=True`
        to correctly load the corresponding embedding files.
    
    low_shot : bool
        for low shot setting

    num_min_samples: int
        number of minority samples in low shot regime
    """

    data_info = DATA_INFO[id_name]
    ood_info = data_info['ood']
    near_datasets = ood_info['near']['datasets']
    far_datasets = ood_info['far']['datasets']

    if 'waterbirds' in id_name:
        dataloader_dict = {}

        if use_features:
            if fine_tuned:
                id_emb_path = f"{finetuned_embeddings_paths}/{id_name}/wb_embs_{model_name}_{correlation}_seed{seed}.npy"
            else:
                id_emb_path = f"{data_info['id']['emb_path']}/wb_embs_{model_name}_{correlation}.npy"
            train_data, val_data, test_data = load_embeddings_and_labels(id_emb_path)

            if low_shot:
                print(f'[Low-Shot] Sampling groups based on correlation {correlation}...')

                emb_dict = np.load(id_emb_path, allow_pickle=True).item()

                group_to_samples = { (0,0): [], (0,1): [], (1,0): [], (1,1): [] }
                for name, emb in emb_dict.items():
                    parts = name.split('_')
                    y = int(parts[0])
                    place = int(parts[1])
                    split = int(parts[2])
                    if split != 0:  # Only train set
                        continue
                    group_to_samples[(y, place)].append((emb, y))

                minority_groups = [(0,1), (1,0)]
                majority_groups = [(0,0), (1,1)]

                # Low-shot sampling
                train_embs, train_labels = [], []
                for group in minority_groups:
                    samples = group_to_samples[group]
                    np.random.seed(seed)
                    print(num_min_samples, len(samples))
                    selected = np.random.choice(len(samples), size=min(num_min_samples, len(samples)), replace=False)
                    for i in selected:
                        emb, label = samples[i]
                        train_embs.append(emb)
                        train_labels.append(label)

                for group in majority_groups:
                    samples = group_to_samples[group]
                    # Compute number of samples to match correlation level (roughly)
                    # If correlation=90, then 90% of total samples should be from majority groups
                    total_minority = len(minority_groups) * num_min_samples
                    total_majority = int(((correlation / 100) / (1 - (correlation / 100))) * total_minority)
                    per_group = total_majority // len(majority_groups)
                    np.random.seed(seed)
                    selected = np.random.choice(len(samples), size=min(per_group, len(samples)), replace=False)
                    for i in selected:
                        emb, label = samples[i]
                        train_embs.append(emb)
                        train_labels.append(label)

                print(f'[Few-Shot] Final train set: {len(train_embs)} samples')
                train_data = (np.array(train_embs), np.array(train_labels))
            
            train_loader, val_loader, test_loader = create_dataloaders(train_data, val_data, test_data, batch_size)
            
            if fine_tuned:
                ood_emb_path = f"{finetuned_embeddings_paths}/ood_embeddings_{model_name}_{id_name}_{correlation}_seed{seed}.npy"

            else:
                ood_emb_path = f"{data_info['ood']['emb_path']}/ood_embeddings_{model_name}.npy"

            dataloader_dict['id'] = {
                    'train': train_loader,
                    'val': val_loader,
                    'test': test_loader
                }

            dataloader_dict['ood'] = {'val': {}, 'near': {}, 'far': {}}
            ood_embeddings = load_embeddings(ood_emb_path)

            def create_ood_split_loaders(emb_dict, dataset_name, split_key):
                emb_list = [emb_dict[k] for k in sorted(emb_dict.keys(), key=lambda k: int(str(k).split('_')[0]))] \
                    if split_key == 'near' else [emb_dict[k] for k in sorted(emb_dict.keys())]

                emb_tensor = torch.tensor(emb_list)
                num_samples = len(emb_tensor)

                np.random.seed(seed)
                all_indices = np.random.permutation(num_samples)
                val_size = int(0.1 * num_samples)

                val_indices = all_indices[:val_size]
                ood_split_indices = all_indices[val_size:]
                ood_split_dataset = Subset(FeatDataset(emb_tensor, torch.zeros(num_samples)), ood_split_indices)
                ood_split_loader = DataLoader(ood_split_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
                dataloader_dict['ood'][split_key][dataset_name] = ood_split_loader

                if (
                    (split_key == 'near' and ood_val_source in ['sp', 'both']) or
                    (split_key == 'far' and ood_val_source in ['nsp', 'both'])
                ):
                    val_dataset = Subset(FeatDataset(emb_tensor, torch.zeros(num_samples)), val_indices)
                    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
                    dataloader_dict['ood']['val'][dataset_name] = val_loader

            for dataset_name, emb_dict in ood_embeddings.items():
                if dataset_name in near_datasets:
                    create_ood_split_loaders(emb_dict, dataset_name, split_key='near')
                elif dataset_name in far_datasets:
                    create_ood_split_loaders(emb_dict, dataset_name, split_key='far')

            return dataloader_dict
            
        else:
            
            dataloader_dict = {}

            id_data_path = f"{data_info['id']['data_dir']}/waterbird_complete{correlation}_forest2water2"
            wb_train_loader, wb_val_loader, wb_test_loader = get_waterbird_loaders(
                path=id_data_path, batch_size=batch_size, use_train_transform=True, seed=seed)

            dataloader_dict['id'] = {
                'train': wb_train_loader,
                'val': wb_val_loader,
                'test': wb_test_loader
            }

            dataloader_dict['ood'] = {'val': {}, 'near': {}, 'far': {}}

            scale = 256.0 / 224.0
            target_resolution = (224, 224)
            large_transform = transforms.Compose([
                transforms.Resize((int(target_resolution[0]*scale), int(target_resolution[1]*scale))),
                transforms.CenterCrop(target_resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

            def create_image_split_loaders(dataset_name, split_key):
                if dataset_name == "SVHN":
                    dataset = SVHN(os.path.join(ood_info['data_dir'], dataset_name), split='test',
                                        transform=large_transform, download=False)
                else:
                    dataset_path = os.path.join(ood_info['data_dir'], dataset_name)
                    dataset = ImageFolder(dataset_path, transform=large_transform)

                num_samples = len(dataset)
                val_size = int(0.1 * num_samples)

                np.random.seed(seed)
                all_indices = np.random.permutation(num_samples)
                val_indices = all_indices[:val_size]
                ood_indices = all_indices[val_size:]

                val_subset = Subset(dataset, val_indices)
                ood_subset = Subset(dataset, ood_indices)

                dataloader_dict['ood'][split_key][dataset_name] = DataLoader(ood_subset, batch_size=batch_size, shuffle=False, num_workers=4)

                if (
                    (split_key == 'near' and ood_val_source in ['sp', 'both']) or
                    (split_key == 'far' and ood_val_source in ['nsp', 'both'])
                ):
                    dataloader_dict['ood']['val'][dataset_name] = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=4)

            for dataset_name in near_datasets:
                create_image_split_loaders(dataset_name, split_key='near')

            for dataset_name in far_datasets:
                create_image_split_loaders(dataset_name, split_key='far')

            return dataloader_dict
        
    elif 'celeba_blond' in id_name:
        dataloader_dict = {}

        data_info = DATA_INFO[id_name]
        ood_info = data_info['ood']
        near_datasets = ood_info['near']['datasets']
        far_datasets = ood_info['far']['datasets']

        if use_features:
            if fine_tuned:
                id_emb_path = f"{finetuned_embeddings_paths}/{id_name}/celeba_embs_{model_name}_{correlation}_seed{seed}.npy"
            else:
                id_emb_path = f"{data_info['id']['emb_path']}/{id_name.split('_')[0]}_embs_{model_name}_{correlation}_seed{seed}.npy"
            train_data, val_data, test_data = load_embeddings_and_labels(id_emb_path)
            if low_shot:
                print(f'[Low-Shot] Sampling groups based on correlation {correlation}...')

                emb_dict = np.load(id_emb_path, allow_pickle=True).item()

                group_to_samples = { (0,0): [], (0,1): [], (1,0): [], (1,1): [] }
                env_dict_reverse = {
                    0: (0, 0),  # 0 -> non-blond hair, female
                    1: (0, 1),  # 1 -> non-blond hair, male
                    2: (1, 0),  # 2 -> blond hair, female
                    3: (1, 1)   # 3 -> blond hair, male
                }
                for name, emb in emb_dict.items():
                    parts = name.split('_')
                    y = int(parts[0])
                    env = int(parts[1])
                    split = int(parts[2])
                    if split != 0:  
                        continue
                    group_to_samples[env_dict_reverse[env]].append((emb, y))

                minority_groups = [(0,1), (1,0)]
                majority_groups = [(0,0), (1,1)]

                train_embs, train_labels = [], []
                for group in minority_groups:
                    samples = group_to_samples[group]
                    np.random.seed(seed)
                    selected = np.random.choice(len(samples), size=min(num_min_samples, len(samples)), replace=False)
                    for i in selected:
                        emb, label = samples[i]
                        train_embs.append(emb)
                        train_labels.append(label)
                if 0 < correlation <= 1:
                    correlation *= 100 
                for group in majority_groups:
                    samples = group_to_samples[group]
                    total_minority = len(minority_groups) * num_min_samples
                    total_majority = int(((correlation / 100) / (1 - (correlation / 100))) * total_minority)
                    per_group = total_majority // len(majority_groups)
                    np.random.seed(seed)
                    selected = np.random.choice(len(samples), size=min(per_group, len(samples)), replace=False)
                    for i in selected:
                        emb, label = samples[i]
                        train_embs.append(emb)
                        train_labels.append(label)

                print(f'[Few-Shot] Final train set: {len(train_embs)} samples')
                train_data = (np.array(train_embs), np.array(train_labels))
            
            
            train_loader, val_loader, test_loader = create_dataloaders(train_data, val_data, test_data, batch_size)
            
            if fine_tuned:
                ood_emb_path = f"{finetuned_embeddings_paths}/ood_embeddings_{model_name}_{id_name}_{correlation}_seed{seed}.npy"
            else:
                ood_emb_path = f"{data_info['ood']['emb_path']}/ood_embeddings_{model_name}.npy"

            dataloader_dict['id'] = {
                    'train': train_loader,
                    'val': val_loader,
                    'test': test_loader
                }

            dataloader_dict['ood'] = {'val': {}, 'near': {}, 'far': {}}
            ood_emb_path = f"{data_info['ood']['emb_path']}/ood_embeddings_{model_name}.npy"
            ood_embeddings = load_embeddings(ood_emb_path)

            def create_ood_split_loaders(emb_dict, dataset_name, split_key):
                emb_list = [emb_dict[k] for k in sorted(emb_dict.keys(), key=lambda k: int(str(k).split('_')[0]))] \
                    if split_key == 'near' else [emb_dict[k] for k in sorted(emb_dict.keys())]

                emb_tensor = torch.tensor(emb_list)
                num_samples = len(emb_tensor)

                np.random.seed(seed)
                all_indices = np.random.permutation(num_samples)
                val_size = int(0.1 * num_samples)

                val_indices = all_indices[:val_size]
                ood_split_indices = all_indices[val_size:]
                ood_split_dataset = Subset(FeatDataset(emb_tensor, torch.zeros(num_samples)), ood_split_indices)
                ood_split_loader = DataLoader(ood_split_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
                dataloader_dict['ood'][split_key][dataset_name] = ood_split_loader

                if (
                    (split_key == 'near' and ood_val_source in ['sp', 'both']) or
                    (split_key == 'far' and ood_val_source in ['nsp', 'both'])
                ):
                    val_dataset = Subset(FeatDataset(emb_tensor, torch.zeros(num_samples)), val_indices)
                    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
                    dataloader_dict['ood']['val'][dataset_name] = val_loader

            for dataset_name, emb_dict in ood_embeddings.items():
                if dataset_name in near_datasets:
                    create_ood_split_loaders(emb_dict, dataset_name, split_key='near')
                elif dataset_name in far_datasets:
                    create_ood_split_loaders(emb_dict, dataset_name, split_key='far')
            print(dataloader_dict)
            return dataloader_dict


    elif 'urbancars' in id_name:
        dataloader_dict = {}

        data_info = DATA_INFO[id_name]
        ood_info = data_info['ood']
        near_datasets = ood_info['near']['datasets']
        far_datasets = ood_info['far']['datasets']

        if use_features:
            id_emb_path = f"{data_info['id']['emb_path']}/urb_embs_{model_name}_{correlation}.npy"
            train_data, val_data, test_data = load_embeddings_and_labels_urb(id_emb_path)
            
            if low_shot:
                print(f'[Low-Shot] Sampling groups based on correlation {correlation}...')

                emb_dict = np.load(id_emb_path, allow_pickle=True).item()

                group_to_samples = {}

                for name, emb in emb_dict.items():
                    parts = name.split('_')
                    if len(parts) < 4:
                        continue
                    label = int(parts[0])
                    place = int(parts[1])       
                    co_occ = int(parts[2])      
                    split = int(parts[3])  

                    if split != 0: 
                        continue

                    group = (label, place, co_occ)
                    if group not in group_to_samples:
                        group_to_samples[group] = []
                    group_to_samples[group].append((emb, label, place, co_occ))

                minority_groups = [(0,0,1), (0,1,0), (0,1,1), (1,0,0), (1,1,0), (1,0,1)]  
                majority_groups = [(0,0,0), (1,1,1)]  

                train_embs, train_labels, train_places, train_cos = [], [], [], []

                for group in minority_groups:
                    samples = group_to_samples.get(group, [])
                    np.random.seed(seed)
                    selected = np.random.choice(len(samples), size=min(num_min_samples, len(samples)), replace=False)
                    for i in selected:
                        emb, label, place, co = samples[i]
                        train_embs.append(emb)
                        train_labels.append(label)
                        train_places.append(place)
                        train_cos.append(co)


                for group in majority_groups:
                    samples = group_to_samples.get(group, [])
                    total_minority = len(minority_groups) * num_min_samples
                    total_majority = int(((correlation / 100) / (1 - (correlation / 100))) * total_minority)
                    per_group = total_majority // len(majority_groups)
                    np.random.seed(seed)
                    selected = np.random.choice(len(samples), size=min(per_group, len(samples)), replace=False)
                    for i in selected:
                        emb, label, place, co = samples[i]
                        train_embs.append(emb)
                        train_labels.append(label)
                        train_places.append(place)
                        train_cos.append(co)

                print(f'[Few-Shot] Final train set: {len(train_embs)} samples')
                train_data = (np.array(train_embs), np.array(train_labels),  np.array(train_places),  np.array(train_cos))

            train_loader, val_loader, test_loader = create_dataloaders(train_data, val_data, test_data, batch_size, is_urb=True)
            

            ood_emb_path = f"{data_info['ood']['emb_path']}/ood_embeddings_{model_name}.npy"

            dataloader_dict['id'] = {
                    'train': train_loader,
                    'val': val_loader,
                    'test': test_loader
                }

            dataloader_dict['ood'] = {'val': {}, 'near': {}, 'far': {}}
            ood_emb_path = f"{data_info['ood']['emb_path']}/ood_embeddings_{model_name}.npy"
            ood_embeddings = load_embeddings(ood_emb_path)

            def create_ood_split_loaders(emb_dict, dataset_name, split_key):
                emb_list = [emb_dict[k] for k in sorted(emb_dict.keys(), key=lambda k: int(str(k).split('_')[0]))] \
                    if split_key == 'near' else [emb_dict[k] for k in sorted(emb_dict.keys())]

                emb_tensor = torch.tensor(emb_list)
                num_samples = len(emb_tensor)

                np.random.seed(seed)
                all_indices = np.random.permutation(num_samples)
                val_size = int(0.1 * num_samples)

                val_indices = all_indices[:val_size]
                ood_split_indices = all_indices[val_size:]
                ood_split_dataset = Subset(FeatDataset(emb_tensor, torch.zeros(num_samples)), ood_split_indices)
                ood_split_loader = DataLoader(ood_split_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
                dataloader_dict['ood'][split_key][dataset_name] = ood_split_loader

                if (
                    (split_key == 'near' and ood_val_source in ['sp', 'both']) or
                    (split_key == 'far' and ood_val_source in ['nsp', 'both'])
                ):
                    val_dataset = Subset(FeatDataset(emb_tensor, torch.zeros(num_samples)), val_indices)
                    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
                    dataloader_dict['ood']['val'][dataset_name] = val_loader

            for dataset_name, emb_dict in ood_embeddings.items():
                if dataset_name in near_datasets:
                    create_ood_split_loaders(emb_dict, dataset_name, split_key='near')
                elif dataset_name in far_datasets:
                    create_ood_split_loaders(emb_dict, dataset_name, split_key='far')

            return dataloader_dict
    elif 'animals_metacoco' in id_name:
        dataloader_dict = {}

        if use_features:
            anml_train_loader, anml_val_loader, anml_test_loader, anml_ood_loader = get_animal_loaders(
                model_name, batch_size, ood_classes,low_shot=low_shot, num_min_samples=num_min_samples, seed=seed
            )

            dataloader_dict['id'] = {
                'train': anml_train_loader,
                'val': anml_val_loader,
                'test': anml_test_loader,
            }

            dataloader_dict['ood'] = {'val': {}, 'near': {}, 'far': {}}

            ood_emb_path = f"{data_info['ood']['emb_path']}/ood_embeddings_{model_name}.npy"
            ood_embeddings = load_embeddings(ood_emb_path)

            num_samples = len(anml_ood_loader.dataset)
            np.random.seed(seed)
            shuffled_indices = np.random.permutation(num_samples)
            val_size = int(0.1 * num_samples)

            val_indices = shuffled_indices[:val_size]
            near_indices = shuffled_indices[val_size:]

            val_subset = torch.utils.data.Subset(anml_ood_loader.dataset, val_indices)
            near_subset = torch.utils.data.Subset(anml_ood_loader.dataset, near_indices)

            if ood_val_source in ['sp', 'both']:
                dataloader_dict['ood']['val']['animals_ood'] = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=4)
            dataloader_dict['ood']['near']['animals_ood'] = DataLoader(near_subset, batch_size=batch_size, shuffle=False, num_workers=4)

            def create_ood_split_loaders(emb_dict, dataset_name, split_key):
                keys_sorted = (
                    sorted(emb_dict.keys(), key=lambda k: int(str(k).split('_')[0]))
                    if split_key == 'near' else sorted(emb_dict.keys())
                )
                emb_tensor = torch.tensor([emb_dict[k] for k in keys_sorted])
                num_samples = len(emb_tensor)

                if dataset_name.lower() == 'svhn':
                    np.random.seed(seed)
                    indices = np.random.permutation(num_samples)[:2000]
                    emb_tensor = emb_tensor[indices]
                    num_samples = len(emb_tensor)

                np.random.seed(seed)
                indices = np.random.permutation(num_samples)
                val_size = int(0.1 * num_samples)
                val_indices = indices[:val_size]
                split_indices = indices[val_size:]

                dataset = FeatDataset(emb_tensor, torch.zeros(num_samples))
                split_subset = Subset(dataset, split_indices)
                dataloader_dict['ood'][split_key][dataset_name] = DataLoader(
                    split_subset, batch_size=batch_size, shuffle=False, num_workers=4
                )

                if (
                    (split_key == 'near' and ood_val_source in ['sp', 'both']) or
                    (split_key == 'far' and ood_val_source in ['nsp', 'both'])
                ):
                    val_subset = Subset(dataset, val_indices)
                    dataloader_dict['ood']['val'][dataset_name] = DataLoader(
                        val_subset, batch_size=batch_size, shuffle=False, num_workers=4
                    )

            for dataset_name, emb_dict in ood_embeddings.items():
                if dataset_name in far_datasets:
                    create_ood_split_loaders(emb_dict, dataset_name, split_key='far')

        return dataloader_dict
    
    elif 'spurious_imagenet' in id_name:
        dataloader_dict = {}
        data_info = DATA_INFO[id_name]
        ood_info = data_info['ood']
        near_datasets = ood_info['near']['datasets']
        far_datasets = ood_info['far']['datasets']

        id_emb_path = f"{data_info['id']['emb_path']}/sp_imagenet_embs_{model_name}_95.npy"
        train_data, val_data, test_data = load_embeddings_and_labels(id_emb_path)

        if low_shot:
            print(f'[Low-Shot] Sampling {num_min_samples} examples per class (seed={seed})...')
            emb_dict = np.load(id_emb_path, allow_pickle=True).item()

            class_to_samples = {}
            for name, emb in emb_dict.items():
                parts = name.split('_') 
                label, split = int(parts[0]), int(parts[1])
                if split != 0:  
                    continue
                class_to_samples.setdefault(label, []).append((emb, label))

            train_embs, train_labels = [], []
            np.random.seed(seed)
            for label, samples in class_to_samples.items():
                selected = np.random.choice(len(samples), size=min(num_min_samples, len(samples)), replace=False)
                for idx in selected:
                    emb, lbl = samples[idx]
                    train_embs.append(emb)
                    train_labels.append(lbl)

            train_data = (np.array(train_embs), np.array(train_labels))
            print(f'[Few-Shot] Sampled {len(train_embs)} training examples from {len(class_to_samples)} classes')

        train_loader, val_loader, test_loader = create_dataloaders(train_data, val_data, test_data, batch_size)

        dataloader_dict['id'] = {
            'train': train_loader,
            'val': val_loader,
            'test': test_loader
        }

        dataloader_dict['ood'] = {'val': {}, 'near': {}, 'far': {}}
        ood_emb_path = f"{data_info['ood']['emb_path']}/ood_embeddings_{model_name}.npy"
        ood_embeddings = np.load(ood_emb_path, allow_pickle=True).item()

        
        def create_ood_split_loaders(emb_dict, dataset_name, split_key):
            if all(isinstance(k, str) and '_' in k for k in emb_dict.keys()):
                emb_list = [emb_dict[k] for k in sorted(emb_dict.keys(), key=lambda k: int(k.split('_')[0]))]
            else:
                emb_list = [emb_dict[k] for k in sorted(emb_dict.keys())]
            emb_tensor = torch.tensor(emb_list)
            num_samples = len(emb_tensor)

            np.random.seed(seed)
            all_indices = np.random.permutation(num_samples)
            val_size = int(0.1 * num_samples)

            val_indices = all_indices[:val_size]
            ood_split_indices = all_indices[val_size:]

            ood_split_dataset = Subset(FeatDataset(emb_tensor, torch.zeros(num_samples)), ood_split_indices)
            ood_split_loader = DataLoader(ood_split_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
            dataloader_dict['ood'][split_key][dataset_name] = ood_split_loader

            if (
                (split_key == 'near' and ood_val_source in ['sp', 'both']) or
                (split_key == 'far' and ood_val_source in ['nsp', 'both'])
            ):
                val_dataset = Subset(FeatDataset(emb_tensor, torch.zeros(num_samples)), val_indices)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
                dataloader_dict['ood']['val'][dataset_name] = val_loader

        for dataset_name, emb_dict in ood_embeddings.items():
            if dataset_name in near_datasets:
                create_ood_split_loaders(emb_dict, dataset_name, split_key='near')
            elif dataset_name in far_datasets:
                create_ood_split_loaders(emb_dict, dataset_name, split_key='far')

        return dataloader_dict

