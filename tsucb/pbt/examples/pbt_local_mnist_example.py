
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


from copy import deepcopy
from typing import NamedTuple, NoReturn
import numpy as np
import matplotlib.pyplot as plt
import ray
from ray.tune.examples.mnist_pytorch import ConvNet, get_data_loaders, train, test
from tqdm import tqdm
from tsucb.pbt.strategies import ExploitUcb, OrigExploitTruncationSelection
from tsucb.pbt.pbt import Member, Population
import os
import ray.tune as tune

import torch


# ========================================================================= #
# SYSTEM                                                                    #
# ========================================================================= #


class ToyHyperParams(NamedTuple):
    learning_rate: float


class TrainMNIST(tune.Trainable):
    def _setup(self, config):
        use_cuda = config.get("use_gpu") and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.train_loader, self.test_loader = get_data_loaders()
        self.model = ConvNet().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=config.get("lr", 0.01),
            momentum=config.get("momentum", 0.9))

    def eval(self):
        return test(self.model, self.test_loader, self.device)

    def _train(self):
        train(self.model, self.optimizer, self.train_loader, device=self.device)
        return {"mean_accuracy": self.eval()}

    def _save(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
        torch.save(self.model.state_dict(), checkpoint_path)
        return checkpoint_path

    def _restore(self, checkpoint_path):
        self.model.load_state_dict(torch.load(checkpoint_path))


# ========================================================================= #
# Member - TOY                                                              #
# ========================================================================= #

CHECKPOINT_MAP = {}


class MemberMnist(Member):

    def __init__(self):
        super().__init__()
        config = dict(
            lr=np.random.uniform(0.001, 0.1),
            momentum=np.random.uniform(0.1, 0.9),
            use_gpu=True,
        )
        self._trainable = TrainMNIST(config=config)

    def _save_theta(self, id):
        CHECKPOINT_MAP[id] = self._trainable.save(f'./checkpoints/{id}')
    def _load_theta(self, id):
        self._trainable.restore(CHECKPOINT_MAP[id])

    def copy_h(self) -> dict:
        return deepcopy(self._trainable.config)
    def _set_h(self, h) -> NoReturn:
        self._trainable = TrainMNIST(config=h)
        # self._trainable.config = h  # I think this is all you need, but am not sure.
    def _explored_h(self, population: 'Population') -> dict:
        return dict(
           lr=np.random.uniform(0.001, 0.1),
           momentum=np.random.uniform(0.1, 0.9),
        )

    def _step(self, options: dict) -> np.ndarray:
        result = self._trainable.train()
        return result
    def _eval(self, options: dict) -> float:
        return self._trainable.eval()

# ============================================================================ #
# PLOTTING                                                                     #
# ============================================================================ #


#
# def make_subplots(h, w, figsize=None):
#     fig, axs = plt.subplots(h, w, figsize=figsize)
#     return fig, np.array(axs).reshape((h, w))
#
# def plot_performance(ax, population, steps, title):
#     for member, color in zip(population, 'brgcmyk'):
#         vals = [step.p for step in member]
#         ax.plot(vals, color=color, lw=0.7)
#     ax.axhline(y=1.2, linestyle='dotted', color='k')
#     ax.set(xlim=[0, steps-1], ylim=[-0.5, 1.31], title=title, xlabel='Step', ylabel='Q')
#
# def plot_theta(ax, population, steps, title):
#     for member, color in zip(population, 'brgcmyk'):
#         x, y = np.array([step.theta[0] for step in member]), np.array([step.theta[1] for step in member])
#         jumps = np.where([step.exploit_id is not None for step in member])[0]
#         x, y = np.insert(x, jumps, np.nan), np.insert(y, jumps, np.nan)
#         ax.plot(x, y, color=color, lw=0.5, zorder=1)
#         ax.scatter(x, y, color=color, s=1, zorder=2)
#
#     ax.set(xlim=[-0.1, 1], ylim=[-0.1, 1], title=title, xlabel='theta0', ylabel='theta1')

def experiment(options, exploiter, n=20, steps=200, exploit=True, explore=True, title=None):
    population = Population([MemberMnist() for i in range(n)], exploiter=exploiter, options=options)
    population.train(steps, exploit=exploit, explore=explore)

    # Calculates the score as the index of the first occurrence greater than 1.18
    # scores = np.array([[h.p for h in m] for m in population])
    # firsts = np.argmax(scores > 1.18, axis=1)
    # firsts[firsts == 0] = scores.shape[1]
    # time_to_converge = np.min(firsts)

    # score = np.max(population.scores)

    # plot_theta(ax_col[0], population, steps=steps, title=title)
    # plot_performance(ax_col[1], population, steps=steps, title=title)
    return np.max(population.scores), np.average(np.array([[h.p for h in m] for m in population]), axis=0)


if __name__ == '__main__':

    ray.init()

    options = {
        "steps": 20,
        "steps_till_ready": 2,
        "exploration_scale": 0.1,
    }

    # REPEAT EXPERIMENT N TIMES
    n, k, repeats = 10, 2, 1
    score, scores = np.zeros(k), np.zeros((k, options['steps']))
    # fig, axs = make_subplots(2, len(scores))

    with tqdm(range(repeats)) as itr:
        for i in itr:
            score_0, scores_0 = experiment(options, OrigExploitTruncationSelection(), n=n, steps=options["steps"], exploit=True, explore=True, title='PBT Trunc Sel')
            score_1, scores_1 = experiment(options, ExploitUcb(subset_mode='top', incr_mode='exploited', normalise_mode='subset'),  n=n, steps=options["steps"], exploit=True, explore=True, title='PBT Ucb Sel')

            score += [score_0, score_1]
            scores += [scores_0, scores_1]

            print(score_0)
            print(scores_0)
            print(score_1)
            print(scores_1)

            # itr.set_description(f'{np.around(scores / (i + 1), 4)} | {np.around(converge_times / (i + 1), 2)}')
    scores /= repeats
    score /= repeats

    print(f'T: {score[0]} | {list(scores[0])}')
    print(f'U: {score[1]} | {list(scores[1])}')

    fig, ax = plt.subplots(1, 1)

    ax.plot(scores[0], label='PBT Trunc Sel')
    ax.plot(scores[1], label='PBT Ucb Sel')
    ax.legend()
    ax.set(title=f'Trunc vs Ucb: {dict(n=n, r=options["steps_till_ready"])}', xlabel='Steps', ylabel='Ave Max Score')

    fig.show()


