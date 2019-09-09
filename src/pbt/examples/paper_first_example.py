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


import random
from typing import NamedTuple

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from pbt.examples.simplex import SimplexNoise
from pbt.pbt import Member, Population


# ============================================================================ #
# SYSTEM                                                                       #
# ============================================================================ #


class ToyHyperParams(NamedTuple):
    coef: np.ndarray
    alpha: float


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
        return (self._t % population.options.get('steps_till_ready', 4) == 0) and (self != max(population, key=lambda m: m._p))

    def _step(self, options: dict) -> np.ndarray:
        dtheta = -2 * self._h.coef * self._theta
        theta = self._theta + self._h.alpha * dtheta
        return theta

    def _eval(self, options: dict) -> float:
        return 1.2 - np.dot(self._theta, self._theta)

    def _exploit(self, population: 'Population') -> 'ToyMember':
        return max(population, key=lambda m: m._p)

    def _explore(self, population: 'Population') -> ToyHyperParams:
        """perturb hyper-parameters with noise from a normal distribution"""
        s = population.options.get('exploration_scale', 0.1)
        return ToyHyperParams(
            self._h.coef + np.random.randn(*self._h.coef.shape) * s,
            # self._h.alpha,
            abs(self._h.alpha + np.random.randn() * s),
        )


class ToyMemberQuantile(ToyMember):

    def __init__(self, *args, quantile=0.2, **kwargs):
        super().__init__(*args, **kwargs)
        self._quantile=quantile

    def _exploit(self, population: 'Population') -> 'ToyMember':
        scored = sorted(population, key=lambda m: m._p, reverse=True)
        top = max(1, self._quantile * len(scored))
        return random.choice(scored[:top])


def ucb1(X_i, n_i, n, C=1):
    return X_i + C * np.sqrt(np.log2(n) / n_i)


class ToyMemberUcb(ToyMember):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._step_count = 0

    def _step(self, options: dict) -> np.ndarray:
        self._step_count += 1
        return super()._step(options)

    def _explore(self, population: 'Population') -> ToyHyperParams:
        self._step_count = 0
        return super()._explore(population)

    def _exploit(self, population: 'Population') -> 'ToyMember':
        max_score = population.options.setdefault('max_score', float('-inf'))
        if self._p > max_score:
            max_score = self._p
            population.options['max_score'] = self._p

        total_steps = max(m._step_count for m in population) * len(population)

        scores = np.array([ucb1(
            X_i=m._p / abs(max_score) if max_score else 0,
            n_i=(m._step_count + 1),
            n=(total_steps + 1),
            C=0.1
        ) for m in population])

        scores[scores < 0] = 0
        scores = np.nan_to_num(scores)
        if sum(scores) == 0:
            scores[:] = 1

        print(scores)

        index = np.argmax(scores)
        choice = population[index]

        # TODO: should it not be this?
        choice = np.random.choice(range(len(population)), p=scores/np.sum(scores))
        choice = population[choice]
        # trials.remove(choice)

        self._step_count = 0
        choice._step_count = 0

        return choice


N = 10
NOISE = SimplexNoise(N)


class SimplexMember(ToyMemberUcb):

    def copy_h(self) -> np.ndarray:
        return np.copy(self._h)

    def copy_theta(self) -> np.ndarray:
        return np.copy(self._theta)

    def _step(self, options: dict) -> np.ndarray:
        # self._theta += self._h * 0.001
        return self._theta

    def _eval(self, options: dict) -> float:
        return abs(NOISE.simplexNoise(list(self._h)))

    def _explore(self, population: 'Population') -> ToyHyperParams:
        """perturb hyper-parameters with noise from a normal distribution"""
        s = population.options.get('exploration_scale', 0.1)
        return self._h + (np.random.random(N) * np.random.randint(0, 2, N) * np.random.randint(0, 2, N)) * s


# ============================================================================ #
# PLOTTING                                                                     #
# ============================================================================ #

options = None


def plot_performance(population, i, steps, title):
    plt.subplot(2, 4, i)

    for member, color in zip(population, 'brgcmyk'):
        vals = [step.p for step in member]
        plt.plot(vals, color=color, lw=0.7)

    plt.axhline(y=1.2, linestyle='dotted', color='k')
    axes = plt.gca()
    axes.set_xlim([0, steps])
    axes.set_ylim([0.0, 1.21])

    plt.title(title)
    plt.xlabel('Step')
    plt.ylabel('Q')


def plot_theta(population, i, steps, title):
    plt.subplot(2, 4, i)

    for member, color in zip(population, 'brgcmyk'):
        x = [step.theta[0] for step in member]
        y = [step.theta[1] for step in member]

        # x = np.convolve(x, np.ones(5) / 5)
        # y = np.convolve(y, np.ones(5) / 5)

        plt.scatter(x, y, color=color, s=0.25)

    axes = plt.gca()
    axes.set_xlim([0, 1])
    axes.set_ylim([0, 1])

    plt.title(title)
    plt.xlabel('theta0')
    plt.ylabel('theta1')


def make_plot(idx, steps=200, exploit=True, explore=True, title=None, clazz=ToyMemberUcb):
    population = Population([
        clazz(ToyHyperParams(np.array([0., 1.]), 0.01), np.array([.9, .9])),
        clazz(ToyHyperParams(np.array([1., 0.]), 0.01), np.array([.9, .9])),
        clazz(ToyHyperParams(np.array([0., 1.]), 0.02), np.array([.9, .9])),
        # clazz(ToyHyperParams(np.array([1., 0.]), 0.02), np.array([.9, .9])),
        # clazz(ToyHyperParams(np.array([0., 1.]), 0.02), np.array([.9, .9])),
        # clazz(ToyHyperParams(np.array([1., 0.]), 0.02), np.array([.9, .9])),
    ], options).train(steps, exploit=exploit, explore=explore)


    score = min(len(m) - sum(1 if h.p > 1.15 else 0 for h in m) for m in population)
    # score = max(max(h.p for h in m) for m in population)
    print(f"{population.best.eval(population.options)}: {title} - {score}")

    plot_theta(population, idx, steps=steps, title=title)
    plot_performance(population, idx+4, steps=steps, title=title)

    return score


if __name__ == '__main__':

    options = {
        "steps": 100,
        "steps_till_ready": 5,
        "exploration_scale": 0.1,
    }

    n, a, b, c = 100, 0, 0, 0
    for i in tqdm(range(n)):
        a += make_plot(1, steps=options["steps"], exploit=True, explore=True, clazz=ToyMemberUcb, title='PBT UCB')
        b += make_plot(2, steps=options["steps"], exploit=True, explore=True, clazz=ToyMemberQuantile, title='PBT Quantile')
        c += make_plot(3, steps=options["steps"], exploit=True, explore=True, clazz=ToyMember, title='PBT Toy')
        print()
    nums = np.array([a, b, c]) / n
    a, b, c = nums # - np.max(nums)

    print('U: {}, Q: {}, T: {}'.format(a, b, c))


    # make_plot(4, steps=options["steps"], exploit=False, explore=False, clazz=ToyMemberUcb, title='Grid Search')

    plt.show()
