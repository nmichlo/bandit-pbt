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



import argparse
import torch
import torch.optim as optim
import numpy as np
import os

from typing import Tuple, NoReturn
from copy import deepcopy
from tsucb.helper.torch.trainable import TorchTrainable
from tsucb.pbt.pbt import Member, Population, IExploiter, Exploiter
from tsucb.pbt.strategies import GeneralisedExploiter, SuggestUniformRandom, ExploitStrategyTruncationSelection

# ========================================================================= #
# MUTATIONS                                                                 #
# ========================================================================= #

def perturb(value, low, high, min, max):
    if np.random.random() < 0.5:
        val = low*value
    else:
        val = high*value
    val = np.clip(val, min, max)
    return val

def uniform_perturb(value, low, high, min, max):
    val = np.random.uniform(low*value, high*value)
    val = np.clip(val, min, max)
    return val

MUTATIONS = {
    'perturb': perturb,
    'uniform_perturb': uniform_perturb,
}

# ========================================================================= #
# MEMBER                                                                    #
# ========================================================================= #

CHECKPOINT_DIR = './checkpoints'
CHECKPOINT_MAP = {}


class MemberTorch(Member):

    def __init__(self, config):
        super().__init__()
        self._trainable = TorchTrainable(config)

    def _save_theta(self, id):
        CHECKPOINT_MAP[id] = os.path.join(CHECKPOINT_DIR, f'checkpoint_{id}.dat')
        self._trainable.save(CHECKPOINT_MAP[id])

    def _load_theta(self, id):
        self._trainable.restore(CHECKPOINT_MAP[id])

    def copy_h(self) -> dict:
        return self._trainable.get_config()
    def _set_h(self, h) -> NoReturn:
        self._trainable = TorchTrainable(config=h)

    def _explored_h(self, population: 'Population') -> dict:
        config = self._trainable.get_config()
        mutations = config['mutations']
        assert len(mutations) > 0
        for path, args in mutations.items():
            current, keys = config, path.split('/')
            for k in keys[:-1]:
                current = current[k]
            current[keys[-1]] = MUTATIONS[args[0]](current[keys[-1]], *args[1:])
        return config

    def _step(self, options: dict) -> NoReturn:
        # TODO this needs to be every N steps, NOT EVERY EPOCH...
        self._trainable.train()
        return None

    def _eval(self, options: dict) -> float:
        results = self._trainable.eval()
        return results['correct']


# ========================================================================= #
# MEMBER                                                                    #
# ========================================================================= #


def main():

    exploiter = GeneralisedExploiter(
        strategy=ExploitStrategyTruncationSelection(),
        suggester=SuggestUniformRandom()
    )

    members = [
        MemberTorch(dict(
            model='example',
            dataset='MNIST',
            loss='MSELoss',

            optimizer='SGD',
            optimizer_options=dict(
                lr=np.random.uniform(0.0001, 0.1),
                momentum=np.random.uniform(0.99, 0.01),
            ),

            mutations={
                'optimizer_options/lr': ('perturb', 0.8, 1.2, 0.0001, 0.1),
                'optimizer_options/momentum': ('perturb', 0.8, 1.2, 0.01, 0.99),
            },

            train_log_interval=-1,
            train_subset=1000,
            batch_size=16,

            use_gpu=True,
        )) for _ in range(10)
    ]

    population = Population(members, exploiter, dict(
        steps_till_ready=1,
        steps=20,
        debug=True,
    ))

    population.train(show_sub_progress=True)


if __name__ == '__main__':
    main()

# ========================================================================= #
# PBT                                                                       #
# ========================================================================= #






# def main():
    # Training settings
    # parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    # parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
    # parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N', help='input batch size for testing (default: 1000)')
    # parser.add_argument('--epochs', type=int, default=50, metavar='N', help='number of epochs to train (default: 10)')
    # parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='learning rate (default: 0.01)')
    # parser.add_argument('--momentum', type=float, default=0.5, metavar='M', help='SGD momentum (default: 0.5)')
    # parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    # parser.add_argument('--seed', type=int, default=42, metavar='S', help='random seed (default: 1)')
    # parser.add_argument('--log-interval', type=int, default=468, metavar='N', help='how many batches to wait before logging training status')
    # parser.add_argument('--save-model', action='store_true', default=False, help='For Saving the current Model')
    # args = parser.parse_args()