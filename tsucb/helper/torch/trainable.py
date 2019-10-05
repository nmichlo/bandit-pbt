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


import torchvision
import tsucb.helper.torch
import torch
import torch.utils.data
import numpy as np
import tsucb.helper.torch.datasets
import tsucb.helper.torch.models
import tsucb.helper as helper

from typing import Tuple, NoReturn
from torch.utils.data import SubsetRandomSampler
from tsucb.helper import util
from copy import deepcopy


# ========================================================================= #
# MAKER HELPER                                                              #
# ========================================================================= #


def _create_loss(name, **kwargs) -> torch.nn.modules.loss._Loss:
    losses = util.get_module_classes(torch.nn.modules.loss, filter_nonlocal=True)
    if name in losses:
        return losses[name](**kwargs)
    else:
        raise KeyError(f'Unsupported loss function: "{name}" Choose one of: {list(losses.keys())}')

def _create_optimizer(name, model_params, **kwargs) -> torch.optim.Optimizer:
    optimizers = util.get_module_classes(torch.optim)
    if name in optimizers:
        if name == 'SGD':  # doesn't have a default for some reason
            kwargs.setdefault('lr',  1e-4)
        return optimizers[name](model_params, **kwargs)
    else:
        raise KeyError(f'Unsupported optimizer: "{name}" Choose one of: {list(optimizers.keys())}')

def _create_dataset(name, **kwargs) -> Tuple[torchvision.datasets.VisionDataset, torchvision.datasets.VisionDataset]:
    allowed = { # datasets with the same options. # 'PhotoTour', 'USPS',
        'CIFAR10',                                             # 32x32x3 # 10
        'CIFAR100',                                            # 32x32x3 # 100
        'MNIST', 'EMNIST', 'FashionMNIST', 'KMNIST', 'QMNIST', # 28x28x1 # 10
    }
    if name in allowed:
        assert not kwargs, 'dataset arguments not allowed'
        return helper.torch.datasets.get_datasets(name)
    else:
        raise KeyError(f'Unsupported dataset: "{name}" Choose one of: {list(allowed)}')

def _create_model(arch, dataset_name, **kwargs) -> torch.nn.Module:
    if arch == 'example':
        assert not kwargs, 'model arguments not allowed'
        if dataset_name in {'MNIST', 'EMNIST', 'FashionMNIST', 'KMNIST', 'QMNIST'}:
            return helper.torch.models.MnistModel()
        else:
            raise KeyError(f'Model: {arch} does not support dataset: {dataset_name}')
    else:
        raise KeyError(f'Unsupported model: "{arch}" Choose one of: {["example"]}')


# ========================================================================= #
# MAKERS                                                                    #
# - THESE SHOULD REMAIN THE SAME TO MATCH THE RAY IMPLEMENTATION            #
# ========================================================================= #


def model_creator(config) -> torch.nn.Module:
    args = config.get('model_options', {})
    return _create_model(config['model'], config['dataset'], **args)

def optimizer_creator(model, config) -> Tuple[torch.nn.modules.loss._Loss, torch.optim.Optimizer]:
    """Returns (criterion/loss, optimizer)"""
    criterion = _create_loss(
        config['loss'],
        **config.get('loss_options', {})
    )

    optimizer_args = config.get('optimizer_options', {})

    optimizer = _create_optimizer(
        config['optimizer'],
        model.parameters(),
        **optimizer_args,
    )
    return criterion, optimizer

def data_creator(config) -> Tuple[torchvision.datasets.VisionDataset, torchvision.datasets.VisionDataset]:
    """Returns (training_set, validation_set)"""
    return _create_dataset(
        config['dataset'],
        **config.get('dataset_options', {})
    )

# ========================================================================= #
# TRAINABLE                                                                 #
# ========================================================================= #

class TorchTrainable(object):
    def __init__(self, config):
        self._config = config

        # GPU SUPPORT
        use_cuda = config['use_gpu'] and torch.cuda.is_available()
        self._device = torch.device("cuda" if use_cuda else "cpu")

        # VARS
        self._model: torch.nn.Module = model_creator(config).to(self._device)
        self._loss, self._optimizer = optimizer_creator(self._model, config)
        trainset, testset = data_creator(config)

        self._train_loader = torch.utils.data.DataLoader(
            trainset,
            batch_size=config['batch_size'],
            shuffle=config.get('shuffle', True),
            num_workers=config.get('num_workers', 2 if use_cuda else 0),
            pin_memory=config.get('pin_memory', use_cuda),
            # **({'sampler': SubsetRandomSampler(indices=np.random.choice(a=np.arange(0, len(trainset)), size=config['train_subset'], replace=False))} if config.get('train_subset', False) else {})
        )

        self._test_loader = torch.utils.data.DataLoader(
            testset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config.get('num_workers', 2 if use_cuda else 0),
            pin_memory=config.get('pin_memory', use_cuda)
        )

    def get_config(self) -> dict:
        return deepcopy(self._config)

    def eval(self) -> dict:
        correct, loss = helper.torch.models.test(self._model, self._device, self._test_loader)
        return dict(
            correct=correct,
            loss=loss
        )

    def train(self) -> NoReturn:
        helper.torch.models.train(self._model, self._device, self._train_loader, self._optimizer, log_time_interval=self._config.get('train_log_interval', -1))

    def save(self, path) -> NoReturn:
        state = dict(
            model=self._model.state_dict(),
            optimizer=self._optimizer.state_dict(),
        )
        torch.save(state, path)

    def restore(self, path) -> NoReturn:
        state = torch.load(path)
        self._model.load_state_dict(state['model'])
        self._optimizer.load_state_dict(state['optimizer'])


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
