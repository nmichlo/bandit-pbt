
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


import ray
from ray.experimental.sgd.tf import TFTrainable, TFTrainer
import tensorflow_datasets as tfds
from tsucb.helper import models


# ========================================================================= #
# trainable                                                                 #
# ========================================================================= #


def _model_creator(config):
    if 'example' == config['model']:
        if 'mnist' == config['dataset']:
            model = models.create_mnist_model('channels_first' if config['use_gpu'] else 'channels_last')
        else:
            raise KeyError('Only dataset="mnist" is currently supported for model="example"')
    else:
        raise KeyError('Only model="example" is currently available.')

    model.compile(
        optimizer=config['optimizer'],
        loss=config['loss'],
        metrics=config.get('metrics', [config['loss']])
    )

    print(model._feed_input_names)

    return model


def _data_creator(config, return_info=False):
    """
    Examples: iris, cifar10, cifar100, mnist, emnist, kmnist, fashion_mnist
    """
    data, info = tfds.load(config['dataset'], with_info=True)
    if return_info:
        return info
    else:
        return data['train'], data['test']


class GeneralTrainable(TFTrainable):

    def __init__(self, config: dict = None, logger_creator=None):
        dataset_info = _data_creator(config, return_info=True)
        print(f'\n[DATASET]:\n{dataset_info}\n')
        # SET BATCH SIZE DEFAULTS
        fit_config = config.setdefault('fit_config', {})
        evaluate_config = config.setdefault('evaluate_config', {})
        fit_config.setdefault('steps_per_epoch', dataset_info.splits['train'].num_examples // config['batch_size'])
        evaluate_config.setdefault('steps', dataset_info.splits['test'].num_examples // config['batch_size'])
        # INITIALISE PARENT
        super().__init__(config=config, logger_creator=logger_creator)

    def _setup(self, config):
        self._trainer = TFTrainer(
            model_creator=_model_creator,
            data_creator=_data_creator,
            config=config,
            num_replicas=config['num_replicas'],
            use_gpu=config['use_gpu']
        )


# ========================================================================= #
# END                                                                       #
# ========================================================================= #


ray.init()

trainable = GeneralTrainable(dict(
    # trainer config
    batch_size=16,
    model='example',
    dataset='mnist',
    optimizer='sgd',
    loss='mse',
    # trainable config
    use_gpu=True,
    num_replicas=1,
))

trainable.train()
