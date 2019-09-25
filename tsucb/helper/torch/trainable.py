
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

#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:


import ray
import torch
from ray.experimental.sgd.pytorch import PyTorchTrainable, PyTorchTrainer
import torch.optim
from helper import util
from helper.torch import datasets


# ========================================================================= #
# HELPER                                                                    #
# ========================================================================= #
from helper.torch.models import create_mnist_model


def create_loss(name, **kwargs):
    losses = util.get_module_classes(torch.nn.modules.loss, filter_nonlocal=True)
    if name in losses:
        return losses[name](**kwargs)
    else:
        raise KeyError(f'Unsupported loss function: "{name}" Choose one of: {list(losses.keys())}')


def create_optimizer(name, params, **kwargs):
    optimizers = util.get_module_classes(torch.optim)
    if name in optimizers:
        if name == 'SGD':  # doesn't have a default for some reason
            kwargs.setdefault('lr',  1e-4)
        return optimizers[name](params, **kwargs)
    else:
        raise KeyError(f'Unsupported optimizer: "{name}" Choose one of: {list(optimizers.keys())}')


def create_dataset(name, **kwargs):
    allowed = { # datasets with the same options. # 'PhotoTour', 'USPS',
        'CIFAR10',                                             # 32x32x3 # 10
        'CIFAR100',                                            # 32x32x3 # 100
        'MNIST', 'EMNIST', 'FashionMNIST', 'KMNIST', 'QMNIST', # 28x28x1 # 10
    }
    if name in allowed:
        assert not kwargs, 'dataset arguments not allowed'
        return datasets.get_datasets(name)
    else:
        raise KeyError(f'Unsupported dataset: "{name}" Choose one of: {list(allowed)}')


def create_model(arch, dataset_name, **kwargs):
    if arch == 'example':
        assert not kwargs, 'model arguments not allowed'
        if dataset_name == 'MNIST':
            return create_mnist_model()
        else:
            raise KeyError(f'Model: {arch} does not support dataset: {dataset_name}')
    else:
        raise KeyError(f'Unsupported model: "{arch}" Choose one of: {["example"]}')


# ========================================================================= #
# trainable                                                                 #
# ========================================================================= #


class GeneralTrainable(PyTorchTrainable):
    def _setup(self, config):

        def _model_creator(config):
            args = config.get('model_options', {})
            return create_model(config['model'], config['dataset'], **args)

        def _optimizer_creator(model, config):
            """Returns (criterion/loss, optimizer)"""
            criterion = create_loss(
                config['loss'],
                **config.get('loss_options', {})
            )

            optimizer_args = config.get('optimizer_options', {})
            optimizer_args.setdefault('lr', 1e-4)

            optimizer = create_optimizer(
                config['optimizer'],
                model.parameters(),
                **optimizer_args,
            )
            return criterion, optimizer

        def _data_creator(config):
            """Returns (training_set, validation_set)"""
            return create_dataset(
                config['dataset'],
                **config.get('dataset_options', {})
            )

        self._trainer = PyTorchTrainer(
            model_creator=_model_creator,
            data_creator=_data_creator,
            optimizer_creator=_optimizer_creator,
            config=config,
            num_replicas=config.get('num_replicas', 1),
            use_gpu=config.get('use_gpu', True),
            batch_size=config.get('batch_size', 16),
            backend=config.get('backend', 'auto')
        )


# ========================================================================= #
# END                                                                       #
# ========================================================================= #


ray.init()

trainable = GeneralTrainable(dict(
    model='example',
    dataset='MNIST',
    optimizer='SGD',
    loss='MSELoss',

    optimizer_options={},
    loss_options={},
    model_options={},
    dataset_options={},
))

trainable.train()
