#  Copyright (c) 2019 Nathan Juraj Michlo
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.


import os
import torch.utils.data
import torchvision
from torchvision import transforms
from tqdm import tqdm
from cachier import cachier, pickle_core as cachier_pickle_core
from helper import util
from filelock import FileLock


# ========================================================================= #
# VARS                                                                      #
# ========================================================================= #


DATA_DIR = os.path.expanduser(os.getenv('DATA_DIR', '~/workspace/data'))
print(f'DATA_DIR="{DATA_DIR}"')

# Change cache directory
cachier_pickle_core.CACHIER_DIR = os.path.join(DATA_DIR, 'cachier')
cachier_pickle_core.EXPANDED_CACHIER_DIR = os.path.expanduser(cachier_pickle_core.CACHIER_DIR)


# ========================================================================= #
# NORMALISATION                                                             #
# ========================================================================= #


@cachier()
def cal_batch_mean_std(dataset_cls):
    # Datasets are not threadsafe
    # with FileLock(f"{DATA_DIR}/data.lock"):
    train_transform = transforms.Compose([transforms.ToTensor()])
    train_set = dataset_cls(root=DATA_DIR, train=True, download=True, transform=train_transform)
    loader = torch.utils.data.DataLoader(train_set, batch_size=16, num_workers=0, shuffle=False)

    n = len(loader.dataset)

    mean, mean2 = 0, 0
    for images, _ in tqdm(loader, f'{dataset_cls.__name__} mean & std'):
        batch_size = images.size(0)  # batch size (the last batch can have smaller size!)
        images = images.view(batch_size, images.size(1), -1)
        # probably not super accurate due to "* (batch_size / n)" as this number is small
        mean += images.mean((0, 2)) * (batch_size / n)
        mean2 += (images ** 2).mean((0, 2)) * (batch_size / n)

    std = (mean2 - mean ** 2) ** 0.5
    return mean.detach().numpy(), std.detach().numpy()

# ========================================================================= #
# LOAD                                                                      #
# ========================================================================= #

def get_dataset_class(dataset_name):
    if type(dataset_name) != str:
        dataset_name = dataset_name.__name__
    assert dataset_name not in {'DatasetFolder', 'VisionDataset', 'ImageFolder'}, 'An abstract dataset was specified'
    datasets = util.get_module_classes(torchvision.datasets)
    assert dataset_name in datasets, f'dataset_name "{dataset_name}" not in {sorted(datasets.keys())}'
    return datasets[dataset_name]

def get_datasets(dataset_name):
    dataset_cls = get_dataset_class(dataset_name)
    # Datasets are not threadsafe
    # with FileLock(f"{DATA_DIR}/data.lock"):
    mean, std = cal_batch_mean_std(dataset_cls)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    trainset = dataset_cls(root=DATA_DIR, train=True, download=True, transform=transform)
    testset = dataset_cls(root=DATA_DIR, train=False, download=True, transform=transform)
    return trainset, testset

def get_dataset_loaders(dataset_name, batch_size=16, shuffle=True, num_workers=1):
    trainset, testset = get_datasets(dataset_name)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return trainloader, testloader

def get_dataset_maker(dataset_name):
    def inner(config):
        return get_dataset_loaders(
            dataset_name,
            batch_size=config.get('batch_size', 16),
            shuffle=config.get('shuffle', True),
            num_workers=config.get('num_workers', 1),
        )
    return inner


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
