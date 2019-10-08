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
from tqdm import tqdm

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
    losses = {k: v for k, v in losses.items() if not k.startswith('_')}
    if name in losses:
        return losses[name](**kwargs)
    else:
        raise KeyError(f'Unsupported loss function: "{name}" Choose one of: [{", ".join(losses.keys())}]')

def _create_optimizer(name, model_params, **kwargs) -> torch.optim.Optimizer:
    optimizers = util.get_module_classes(torch.optim)
    optimizers = {k: v for k, v in optimizers.items() if not k.startswith('_')}
    if name in optimizers:
        if name == 'SGD':  # doesn't have a default for some reason
            kwargs.setdefault('lr',  1e-4)
        return optimizers[name](model_params, **kwargs)
    else:
        raise KeyError(f'Unsupported optimizer: "{name}" Choose one of: [{", ".join(optimizers.keys())}]')

def _create_dataset(name, **kwargs) -> Tuple[torchvision.datasets.VisionDataset, torchvision.datasets.VisionDataset]:
    allowed = { # datasets with the same member_options. # 'PhotoTour', 'USPS',
        'CIFAR10',                                             # 32x32x3 # 10
        'CIFAR100',                                            # 32x32x3 # 100
        'MNIST', 'EMNIST', 'FashionMNIST', 'KMNIST', 'QMNIST', # 28x28x1 # 10
    }
    if name in allowed:
        assert not kwargs, 'dataset arguments not allowed'
        return helper.torch.datasets.get_datasets(name)
    else:
        raise KeyError(f'Unsupported dataset: "{name}" Choose one of: [{", ".join(allowed)}]')

def _create_model(arch, dataset_name, **kwargs) -> torch.nn.Module:
    if arch == 'example':
        assert not kwargs, 'model arguments not allowed'
        if dataset_name in {'MNIST', 'EMNIST', 'FashionMNIST', 'KMNIST', 'QMNIST'}:
            return helper.torch.models.MnistModel()
        else:
            raise KeyError(f'Model: {arch} does not support dataset: {dataset_name}')
    else:
        raise KeyError(f'Unsupported model: "{arch}" Choose one of: [{", ".join(["example"])}]')


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

# TODO: move into population
_LOADER_STORAGE = {}


class TorchTrainable(object):

    _DEFAULT_WORKERS = 4
    _DEFAULT_TRAIN_SHUFFLE = True

    def __init__(self, config, share_id=None):
        self._share_id = share_id

        # GPU SUPPORT
        self._use_gpu = config['use_gpu'] and torch.cuda.is_available()
        self._device = torch.device("cuda" if self._use_gpu else "cpu")

        # CONFIG CONSTS
        self._train_images_per_step = config.get('train_images_per_step', None)
        self._batch_size = config['batch_size']
        self._train_shuffle = config.get('train_shuffle', self._DEFAULT_TRAIN_SHUFFLE)
        self._num_workers = config.get('num_workers', self._DEFAULT_WORKERS if self._use_gpu else 0)
        self._pin_memory = config.get('pin_memory', False) and self._use_gpu

        # TORCH VARS
        self._model = None
        self._criterion = None
        self._optimizer = None
        self._train_loader = None
        self._test_loader = None

        # VARS
        self._trainer = None

        # INIT
        self.reset(config)

    def _check_config(self, config):
        assert self._train_images_per_step == config.get('train_images_per_step', None),                      'Changes to dataset "train_images_per_step" not allowed during training.'
        assert self._batch_size == config['batch_size'],                                                      'Changes to dataset "batch_size" not allowed during training.'
        assert self._train_shuffle == config.get('train_shuffle', self._DEFAULT_TRAIN_SHUFFLE),               'Changes to dataset "train_shuffle" not allowed during training.'
        assert self._num_workers == config.get('num_workers', self._DEFAULT_WORKERS if self._use_gpu else 0), 'Changes to dataset "num_workers" not allowed during training.'
        assert self._pin_memory == config.get('pin_memory', False) and self._use_gpu,                                   'Changes to dataset "pin_memory" not allowed during training.'

    def reset(self, config):
        if (self._train_loader is None) and (self._test_loader is None) and (self._trainer is None):
            # TODO: MUTATION CHECKS
            self._model = model_creator(config).to(self._device)

            # TRAINER
            # TODO: fix train region
            self._trainer = helper.torch.models.StepTrainer()

            # INIT DATASET
            if (self._share_id is None) or (self._share_id not in _LOADER_STORAGE):
                trainset, testset = data_creator(config)
                self._train_loader = torch.utils.data.DataLoader(
                    trainset,
                    batch_size=self._batch_size,
                    shuffle=self._train_shuffle,
                    num_workers=self._num_workers,
                    pin_memory=self._pin_memory,
                )
                self._test_loader = torch.utils.data.DataLoader(
                    testset,
                    batch_size=self._batch_size,
                    shuffle=False,
                    num_workers=self._num_workers,
                    pin_memory=self._pin_memory,
                )

                if self._share_id is not None:
                    _LOADER_STORAGE[self._share_id] = (self._train_loader, self._test_loader)

            # DATASET
            if self._share_id is not None:
                self._train_loader, self._test_loader = _LOADER_STORAGE[self._share_id]
        else:
            self._check_config(config)

        # LOSS & OPTIMIZER
        self._criterion, self._optimizer = optimizer_creator(self._model, config)

    def cleanup(self):
        if self._share_id is not None:
            if self._share_id in _LOADER_STORAGE:
                del _LOADER_STORAGE[self._share_id]
        self._model = None
        self._criterion = None
        self._optimizer = None
        self._train_loader = None
        self._test_loader = None
        self._trainer = None

    def eval(self) -> dict:
        correct, loss = helper.torch.models.test(self._model, self._device, self._test_loader, self._criterion)
        return dict(
            correct=correct,
            loss=loss
        )

    def train(self) -> NoReturn:
        # helper.torch.models.train(self._model, self._device, self._train_loader, self._optimizer) #, log_time_interval=self._config.get('train_log_interval', -1))
        self._trainer.train(self._model, self._device, self._train_loader, self._optimizer, self._criterion, num_images=self._train_images_per_step)

    def save(self, path) -> NoReturn:
        state = dict(
            model=self._model.state_dict(),
            # optimizer=self._optimizer.state_dict(),
        )
        torch.save(state, path)

    def restore(self, path) -> NoReturn:
        state = torch.load(path)
        self._model.load_state_dict(state['model'])
        # self._optimizer.load_state_dict(state['optimizer'])


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
