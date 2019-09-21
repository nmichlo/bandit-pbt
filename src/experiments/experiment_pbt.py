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


import atexit
import random
import settings
import comet_ml
from pprint import pprint
import numpy as np
import torch
import copy


# ========================================================================= #
# COMET ML                                                                  #
# ========================================================================= #

# Settings are automatically read from environment variables.
# https://www.comet.ml/docs/python-sdk/advanced/#comet-configuration-variables
from pbt.pbt import Member
from pbt.strategies import ExploitUcb, ExploitTruncationSelection

EXP = comet_ml.Experiment(
    disabled=(not settings.ENABLE_COMET_ML),
)

if EXP.alive is False:
    raise Exception("Comet.ml isn't alive!")


# ========================================================================= #
# SCHEDULER                                                                 #
# ========================================================================= #


if settings.EXP_EXPLOITER == 'ts-ucb':
    exploiter = ExploitUcb
elif settings.EXP_EXPLOITER == 'ts':
    exploiter = ExploitTruncationSelection
else:
    raise KeyError(f'Invalid scheduler specified: {settings.EXP_SCHEDULER}')


# ========================================================================= #
# MEMBER                                                                    #
# ========================================================================= #


CHECKPOINT_MAP = {}


class ModelTrainer(Member):

    def __init__(self):
        super().__init__()
        self._last_eval = None

        self._model: torch.nn.Module = None

    def copy_h(self) -> dict:
        return copy.deepcopy(self._h)

    def _save_theta(self, id):
        path = f'./checkpoints/pbt-member-checkpoint-{id}.tar'
        torch.save({
            'state_dict': self._model.state_dict(),
            'optimizer_dict' : self._optimizer.state_dict(),
        }, path)
        CHECKPOINT_MAP[id] = path

    def _load_theta(self, id):
        checkpoint = torch.load(CHECKPOINT_MAP[id])
        self._model.load_state_dict(checkpoint['state_dict'])
        self._optimizer.load_state_dict(checkpoint['optimizer_dict'])

    def _is_ready(self, population: 'Population') -> bool:
        return self._t % population.options.get('steps_till_ready', 3) == 0  # and (self != max(population, key=lambda m: m._p))

    def _step(self, options: dict) -> np.ndarray:
        result = self._trainable._train()
        self._last_eval = result['mean_accuracy']
        return result

    def _eval(self, options: dict) -> float:
        model.eval()

        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

        return self._last_eval

    def _explore(self, population: 'Population') -> dict:
        return dict(
           lr=np.random.uniform(0.001, 0.1),
           momentum=np.random.uniform(0.1, 0.9),
        )






analysis = tune.run(
    TrainMNIST,
    scheduler=scheduler,
    resources_per_trial=dict(
        cpu=settings.CPUS_PER_NODE,
        gpu=settings.USE_GPU
    ),
    num_samples=settings.EXP_POPULATION_SIZE,
    # compares to values returned from train()
    stop=dict(
        mean_accuracy=0.99,
        training_iteration=20,
    ),
    # sampling functions
    config=dict(
        lr=tune.uniform(0.001, 0.1),
        momentum=tune.uniform(0.1, 0.9),
    ),
)


print(f'Best config is: {analysis.get_best_config(metric="mean_accuracy")}')
print(f'All the configs are:')
pprint(analysis.get_all_configs())


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
