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
from ray.experimental.sgd.pytorch import PyTorchTrainable, PyTorchTrainer
from tsucb.helper.torch.trainable import data_creator, model_creator, optimizer_creator

# ========================================================================= #
# trainable                                                                 #
# ========================================================================= #


class GeneralTrainableRay(PyTorchTrainable):
    def _setup(self, config):
        self._trainer = PyTorchTrainer(
            model_creator=model_creator,
            data_creator=data_creator,
            optimizer_creator=optimizer_creator,
            config=config,
            num_replicas=config.get('num_replicas', 1),
            use_gpu=config.get('use_gpu', True),
            batch_size=config.get('batch_size', 1),
            backend=config.get('backend', 'auto')
        )


# ========================================================================= #
# END                                                                       #
# ========================================================================= #


ray.init()

trainable = GeneralTrainableRay(dict(
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
