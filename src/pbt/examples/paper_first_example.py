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

from pbt.examples.simplex import SimplexNoise
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
            self._h.coef + np.random.normal(0, s, size=self._h.coef.shape),
            self._h.alpha + np.random.normal(0, s),
        )

# ============================================================================ #
# PLOTTING                                                                     #
# ============================================================================ #

options = None


def plot_performance(population, i, steps, title):
    plt.subplot(2, 2, i)

    for member, color in zip(population, 'brgcmyk'):
        vals = [step.p for step in member]
        plt.plot(vals, color=color, lw=0.7)

    plt.axhline(y=1.2, linestyle='dotted', color='k')
    axes = plt.gca()

    axes.set_xlim([0, steps])
    axes.set_ylim([0, 1.21])

    plt.title(title)
    plt.xlabel('Step')
    plt.ylabel('Q')


def plot_theta(population, i, steps, title):
    plt.subplot(2, 2, i)

    for member, color in zip(population, 'brgcmyk'):
        x = np.array([step.theta[0] for step in member])
        y = np.array([step.theta[1] for step in member])
        jumps = np.where([step.exploit_id is not None for step in member])[0]

        x = np.insert(x, jumps, np.nan)
        y = np.insert(y, jumps, np.nan)
        plt.plot(x, y, color=color, lw=0.5, zorder=1)
        plt.scatter(x, y, color=color, s=1, zorder=2)

    axes = plt.gca()
    axes.set_xlim([0, 1])
    axes.set_ylim([0, 1])

    plt.title(title)
    plt.xlabel('theta0')
    plt.ylabel('theta1')


def make_plot(idx, options, exploiter, steps=200, exploit=True, explore=True, title=None):
    population = Population([
        ToyMember(ToyHyperParams(np.array([1., 0.]), 0.01), np.array([.9, .9])),
        ToyMember(ToyHyperParams(np.array([0., 1.]), 0.01), np.array([.9, .9])),
    ], exploiter=exploiter, options=options)

    population.train(steps, exploit=exploit, explore=explore)

    score = max(m.score for m in population)

    # score = min(len(m) - sum(1 if h.p > 1.15 else 0 for h in m) for m in population)
    # score = max(max(h.p for h in m) for m in population)
    # print(f"{population.best.eval(population.options)}: {title} - {score}")

    plot_theta(population, idx, steps=steps, title=title)
    plot_performance(population, idx+2, steps=steps, title=title)

    return score


if __name__ == '__main__':

    options = {
        "steps": 100,
        "steps_till_ready": 5,
        "exploration_scale": 0.1,
    }

    # REPEAT EXPERIMENT N TIMES
    n, scores = 1, np.zeros(2)
    for i in tqdm(range(n)):
        scores[0] += make_plot(1, options, ExploitTruncationSelection(), steps=options["steps"], exploit=False, explore=True, title='PBT Trunc Sel')
        scores[1] += make_plot(2, options, ExploitUcb(),                 steps=options["steps"], exploit=False, explore=True, title='PBT Ucb Sel')
        # print()
    scores /= n

    print('T: {}, U: {}'.format(*scores))

    plt.show()


























