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


from typing import NamedTuple

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from pbt.strategies import ExploitUcb, ExploitTruncationSelection
from pbt.pbt import Member, Population


# ========================================================================= #
# SYSTEM                                                                    #
# ========================================================================= #


class ToyHyperParams(NamedTuple):
    coef: np.ndarray
    alpha: float


# ========================================================================= #
# Member - TOY                                                              #
# ========================================================================= #


class ToyMember(Member):
    """
    options provided:
    - steps_till_ready=4
    - learning_rate=0.01
    - exploration_scale=0.1
    """

    def copy_h(self) -> ToyHyperParams:
        return ToyHyperParams(np.copy(self._h.coef), self._h.alpha)

    def copy_theta(self) -> np.ndarray:
        return np.copy(self._theta)

    def _is_ready(self, population: 'Population') -> bool:
        return self._t % population.options.get('steps_till_ready', 4) == 0  # and (self != max(population, key=lambda m: m._p))

    def _step(self, options: dict) -> np.ndarray:
        dtheta = -2 * self._h.coef * self._theta
        theta = self._theta + self._h.alpha * dtheta
        return theta

    def _eval(self, options: dict) -> float:
        return 1.2 - np.dot(self._theta, self._theta)

    def _explore(self, population: 'Population') -> ToyHyperParams:
        """perturb hyper-parameters with noise from a normal distribution"""
        s = population.options.get('exploration_scale', 0.1)
        return ToyHyperParams(
            np.clip(np.random.normal(self._h.coef, s), 0, 1),
            np.clip(np.random.normal(self._h.alpha, s), 0, 1),
        )

# ============================================================================ #
# PLOTTING                                                                     #
# ============================================================================ #


def make_subplots(h, w, figsize=None):
    fig, axs = plt.subplots(h, w, figsize=figsize)
    return fig, np.array(axs).reshape((h, w))

def plot_performance(ax, population, steps, title):
    for member, color in zip(population, 'brgcmyk'):
        vals = [step.p for step in member]
        ax.plot(vals, color=color, lw=0.7)
    ax.axhline(y=1.2, linestyle='dotted', color='k')
    ax.set(xlim=[0, steps-1], ylim=[-0.5, 1.31], title=title, xlabel='Step', ylabel='Q')

def plot_theta(ax, population, steps, title):
    for member, color in zip(population, 'brgcmyk'):
        x, y = np.array([step.theta[0] for step in member]), np.array([step.theta[1] for step in member])
        jumps = np.where([step.exploit_id is not None for step in member])[0]
        x, y = np.insert(x, jumps, np.nan), np.insert(y, jumps, np.nan)
        ax.plot(x, y, color=color, lw=0.5, zorder=1)
        ax.scatter(x, y, color=color, s=1, zorder=2)

    ax.set(xlim=[-0.1, 1], ylim=[-0.1, 1], title=title, xlabel='theta0', ylabel='theta1')

def make_plot(ax_col, options, exploiter, steps=200, exploit=True, explore=True, title=None):
    population = Population([
        ToyMember(ToyHyperParams(np.array([1., .0]), 0.01), np.array([.9, .9])),
        ToyMember(ToyHyperParams(np.array([.0, 1.]), 0.01), np.array([.9, .9])),
        *[ToyMember(ToyHyperParams(np.array([1., np.random.rand()*0.5]), 0.01), np.array([.9, .9])) for i in range(4)],
        *[ToyMember(ToyHyperParams(np.array([np.random.rand()*0.5, 1.]), 0.01), np.array([.9, .9])) for i in range(4)],
    ], exploiter=exploiter, options=options)

    population.train(steps, exploit=exploit, explore=explore)

    # Calculates the score as the index of the first occurrence greater than 1.18
    scores = np.array([[h.p for h in m] for m in population])
    firsts = np.argmax(scores > 1.18, axis=1)
    firsts[firsts == 0] = scores.shape[1]
    time_to_converge = np.min(firsts)

    score = np.max(population.scores)

    plot_theta(ax_col[0], population, steps=steps, title=title)
    plot_performance(ax_col[1], population, steps=steps, title=title)
    return score, time_to_converge


if __name__ == '__main__':

    options = {
        "steps": 100,
        "steps_till_ready": 1,
        "exploration_scale": 0.1,
    }

    # REPEAT EXPERIMENT N TIMES
    n, scores, converge_times = 10000, np.zeros(2), np.zeros(2)
    fig, axs = make_subplots(2, len(scores))

    with tqdm(range(n)) as itr:
        for i in itr:
            score_0, converge_time_0 = make_plot(axs[:, 0], options, ExploitTruncationSelection(), steps=options["steps"], exploit=True, explore=True, title='PBT Trunc Sel')
            score_1, converge_time_1 = make_plot(axs[:, 1], options, ExploitUcb(),                 steps=options["steps"], exploit=True, explore=True, title='PBT Ucb Sel')
            scores += [score_0, score_1]
            converge_times += [converge_time_0, converge_time_1]
            itr.set_description(f'{np.around(scores / (i + 1), 4)} | {np.around(converge_times / (i + 1), 2)}')
    scores /= n

    print('T: {}, U: {}'.format(*scores))

    # fig.show()


