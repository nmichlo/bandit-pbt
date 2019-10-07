import time
from typing import NamedTuple
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from tsucb.helper import util
from tsucb.pbt.strategies import *
from tsucb.pbt.pbt import Member, Population
import scipy.stats


# ========================================================================= #
# SYSTEM                                                                    #
# ========================================================================= #


class ToyHyperParams(NamedTuple):
    coef: np.ndarray
    alpha: float


# ========================================================================= #
# Member - TOY                                                              #
# ========================================================================= #

_THETA_STORE = {}

class ToyMember(Member):
    """
    options provided:
    - steps_till_ready=4
    - learning_rate=0.01
    - exploration_scale=0.1
    """

    def _setup(self, h, theta):
        self._theta = theta
        self._h = h

    def _save_theta(self, id):
        _THETA_STORE[id] = np.copy(self._theta)
    def _load_theta(self, id):
        self._theta = np.copy(_THETA_STORE[id])

    def copy_h(self) -> ToyHyperParams:
        return ToyHyperParams(np.copy(self._h.coef), self._h.alpha)
    def _set_h(self, h):
        self._h = h
    def _explored_h(self, population: 'Population') -> ToyHyperParams:
        """perturb hyper-parameters with noise from a normal distribution"""
        s = population.options.get('exploration_scale', 0.1)
        return ToyHyperParams(
            np.clip(np.random.normal(self._h.coef, s), 0, 1),
            np.clip(np.random.normal(self._h.alpha, s), 0, 1),
        )

    @property
    def mutable_str(self) -> str:
        return f'coef={self._h.coef}, alpha={self._h.alpha}'

    def _step(self, options: dict) -> dict:
        dtheta = -2 * self._h.coef * self._theta
        self._theta += self._h.alpha * dtheta
        return {'theta': np.copy(self._theta)}
    def _eval(self, options: dict) -> float:
        return 1.2 - np.dot(self._theta, self._theta)

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
        steps = [step for step in member if step.result]
        x, y = np.array([step.result['theta'][0] for step in steps]), np.array([step.result['theta'][1] for step in steps])
        jumps = np.where([step.exploit_id is not None for step in steps])[0]
        x, y = np.insert(x, jumps, np.nan), np.insert(y, jumps, np.nan)
        ax.plot(x, y, color=color, lw=0.5, zorder=1)
        ax.scatter(x, y, color=color, s=1, zorder=2)

    ax.set(xlim=[-0.1, 1], ylim=[-0.1, 1], title=title, xlabel='theta0', ylabel='theta1')

def make_plot(ax_col, options, exploiter, steps=200, exploit=True, explore=True, title=None):
    population = Population([
        # ToyMember(ToyHyperParams(np.array([1., .0]), 0.01), np.array([.9, .9])),
        # ToyMember(ToyHyperParams(np.array([.0, 1.]), 0.01), np.array([.9, .9])),
        *[ToyMember(h=ToyHyperParams(np.random.rand(2) * 0.5, 0.01), theta=np.array([.9, .9])) for i in range(options['population_size'])],
        # *[ToyMember(ToyHyperParams(np.array([np.random.rand()*0.5, 1.]), 0.01), np.array([.9, .9])) for i in range(3)],
    ], exploiter=exploiter, options=options)

    t0 = time.time()
    population.train(steps, exploit=exploit, explore=explore, show_progress=False)
    t1 = time.time()

    # Calculates the score as the index of the first occurrence greater than 1.18
    scores = np.array([[h.p for h in m.history] for m in population])
    firsts = np.argmax(scores > 1.18, axis=1)
    firsts[firsts == 0] = scores.shape[1]
    time_to_converge = np.min(firsts)

    score = np.max(population.scores)

    # plot_theta(ax_col[0], population, steps=steps, title=title)
    # plot_performance(ax_col[1], population, steps=steps, title=title)
    return score, time_to_converge, scores.max(axis=0), len(population), t1 - t0


# ========================================================================== #
# SAME AS PBT PAPER                                                          #
# ========================================================================== #


def run_dual_test():

    options = {
        "repeats": 1000,
        "steps": 11,
        "steps_till_ready": 2,
        "exploration_scale": 0.1,
        "population_size": 50,

        "warn_exploit_self": True,
        "print_scores": False,
    }

    make_exploit_strategy = lambda: ExploitStrategyTruncationSelection()
    # make_exploit_strategy = lambda: ExploitStrategyBinaryTournament()

    # EXPLOITERS
    exploiters = [
        # orig
        # ('orig-ts', lambda: OrigExploitTruncationSelection()),
        # ('orig-ts-eg', lambda: OrigExploitEGreedy(epsilon=0.5, subset_mode='top')),
        # ('orig-ts-ucb', lambda: OrigExploitUcb(c=1.0, subset_mode='top', normalise_mode='subset', incr_mode='exploited')),
        # ('orig-ts-sm', lambda: OrigExploitSoftmax(temperature=1.0, subset_mode='top')),
        # ('orig-ts-esm', lambda: OrigExploitESoftmax(epsilon=0.5, temperature=1.0, subset_mode='top')),
        # new
        ('ts',         lambda: GeneralisedExploiter(make_exploit_strategy(), SuggestUniformRandom())),
        ('ts-egr',     lambda: GeneralisedExploiter(make_exploit_strategy(), SuggestEpsilonGreedy(epsilon=0.75))),
        ('ts-sm',      lambda: GeneralisedExploiter(make_exploit_strategy(), SuggestSoftmax(temperature=1.0))),
        ('ts-esm',     lambda: GeneralisedExploiter(make_exploit_strategy(), SuggestEpsilonSoftmax(epsilon=0.75, temperature=1.0))),
        # ('ts-ucb-0.1',     lambda: GeneralisedExploiter(make_exploit_strategy(), SuggestUcb(c=0.1))),
        ('ts-ucb-0.5',     lambda: GeneralisedExploiter(make_exploit_strategy(), SuggestUcb(c=0.5))),
        ('ts-ucb-1.0',     lambda: GeneralisedExploiter(make_exploit_strategy(), SuggestUcb(c=1.0))),
        ('ts-ucb-2.0',     lambda: GeneralisedExploiter(make_exploit_strategy(), SuggestUcb(c=2.0))),
        # ('ts-eucb',    lambda: GeneralisedExploiter(make_exploit_strategy(), SuggestEpsilonUcb(epsilon=0.5, c=1.0))),
    ]
    k = len(exploiters)

    # RESULTS
    scores, converge_times, scores_per_steps, times = [], [], np.zeros((k, options['steps'])), []
    fig, axs = make_subplots(2, k)

    @util.min_time_elapsed(0.5)
    def print_results(i, n):
        tqdm.write(f"{i:{len(str(n))}d}/{n} {' | '.join(f'{name}: {s:9.7f} {c:9.7f} {t:5.3f}s' for (name, _), s, c, t in zip(exploiters, np.average(scores, axis=0), np.average(converge_times, axis=0), np.average(times, axis=0)))}")

    # EXPERIMENTS
    with tqdm(range(options['repeats'])) as itr:
        for i in itr:
            results = []  # [(score, converge_time, score_seq, pop_len)]
            for name, make_exploiter in exploiters:
                exploiter = make_exploiter()
                result = make_plot(axs[:, 0], options, exploiter=exploiter,  steps=options["steps"], exploit=True, explore=True, title=f'PBT {name}')
                results.append(result)
            r_scores, r_conv_time, r_score_seq, r_pop_sizes, t_times = zip(*results)

            scores.append(r_scores)
            converge_times.append(r_conv_time)
            scores_per_steps += np.array(r_score_seq) / options['repeats']
            times.append(t_times)

            assert all(a == b for a, b in zip(r_pop_sizes[:-1], r_pop_sizes[1:]))

            print_results(i, options['repeats'])


    scores, converge_times, scores_per_steps = np.array(scores), np.array(converge_times), np.array(scores_per_steps)
    fig.show()

    fig, ((ax,),) = make_subplots(1, 1)
    for (name, _), score_per_step in zip(exploiters, scores_per_steps):
        ax.plot(score_per_step, label=f'{name}')
    ax.legend()
    fig.show()


if __name__ == '__main__':
    run_dual_test()


# ========================================================================== #
# END                                                                        #
# ========================================================================== #

