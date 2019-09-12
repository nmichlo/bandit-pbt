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


import torch.utils.data
import torchvision
from torchvision import transforms
import numpy as np
from tqdm import tqdm
from cachier import cachier


# ========================================================================= #
# VARS                                                                      #
# ========================================================================= #


DATA_DIR = '~/workspace/data'


# ========================================================================= #
# NORMALISATION                                                             #
# ========================================================================= #


@cachier()
def cal_batch_mean_std(dataset_cls):
    train_transform = transforms.Compose([transforms.ToTensor()])
    train_set = dataset_cls(root=DATA_DIR, train=True, download=True, transform=train_transform)
    loader = torch.utils.data.DataLoader(train_set, batch_size=11, num_workers=0, shuffle=False)

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


def get_datasets(dataset_cls):
    mean, std = cal_batch_mean_std(dataset_cls)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    trainset = dataset_cls(root=DATA_DIR, train=True, download=True, transform=transform)
    testset = dataset_cls(root=DATA_DIR, train=False, download=True, transform=transform)
    return trainset, testset


def get_dataset_loaders(dataset_cls, batch_size=16, shuffle=True, num_workers=1):
    trainset, testset = get_datasets(dataset_cls)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return trainloader, testloader


def get_dataset_class_names(dataset_cls):
    CLASS_NAMES = {
        torchvision.datasets.CIFAR10: ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    }
    if dataset_cls not in CLASS_NAMES:
        raise KeyError(f'Dataset is currently unknown: {dataset_cls.__name__}')
    return CLASS_NAMES[dataset_cls]


# ========================================================================= \#
# END                                                                       \#
# ========================================================================= \#
