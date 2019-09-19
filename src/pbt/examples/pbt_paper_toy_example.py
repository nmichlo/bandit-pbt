import comet_ml
import pickle
from pprint import pprint
from uuid import uuid4
from typing import NamedTuple
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

import util
from pbt.strategies import ExploitUcb, ExploitTruncationSelection
from pbt.pbt import Member, Population
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

    def __init__(self, h, theta):
        super().__init__()
        self._theta = theta
        self._h = h

    def _save_theta(self, id):
        _THETA_STORE[id] = np.copy(self._theta)

    def _load_theta(self, id):
        self._theta = np.copy(_THETA_STORE[id])

    def copy_h(self) -> ToyHyperParams:
        return ToyHyperParams(np.copy(self._h.coef), self._h.alpha)

    def _is_ready(self, population: 'Population') -> bool:
        return self._t % population.options.get('steps_till_ready', 4) == 0  # and (self != max(population, key=lambda m: m._p))

    def _step(self, options: dict) -> dict:
        dtheta = -2 * self._h.coef * self._theta
        self._theta += self._h.alpha * dtheta
        return { 'theta': np.copy(self._theta) }

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
        *[ToyMember(h=ToyHyperParams(np.random.rand(2) * 0.5, 0.01), theta=np.array([.9, .9])) for i in range(10)],
        # *[ToyMember(ToyHyperParams(np.array([np.random.rand()*0.5, 1.]), 0.01), np.array([.9, .9])) for i in range(3)],
    ], exploiter=exploiter, options=options)

    population.train(steps, exploit=exploit, explore=explore, show_progress=False)

    # Calculates the score as the index of the first occurrence greater than 1.18
    scores = np.array([[h.p for h in m] for m in population])
    firsts = np.argmax(scores > 1.18, axis=1)
    firsts[firsts == 0] = scores.shape[1]
    time_to_converge = np.min(firsts)

    score = np.max(population.scores)

    plot_theta(ax_col[0], population, steps=steps, title=title)
    plot_performance(ax_col[1], population, steps=steps, title=title)
    return score, time_to_converge, scores.max(axis=0), len(population)


def run_dual_test():

    options = {
        "steps": 45,
        "steps_till_ready": 2,
        "exploration_scale": 0.1,
    }

    # REPEAT EXPERIMENT N TIMES
    n = 150

    k, pop_size = 2, None
    scores, converges, score_seq = [], [], np.zeros((k, options['steps']))

    # LOGGING
    fig, axs = make_subplots(2, k)

    with tqdm(range(n)) as itr:
        for i in itr:
            score_0, converge_time_0, score_seq_0, pop_len0 = make_plot(axs[:, 0], options, ExploitTruncationSelection, steps=options["steps"], exploit=True, explore=True, title='PBT Trunc Sel')
            score_1, converge_time_1, score_seq_1, pop_len1 = make_plot(axs[:, 1], options, ExploitUcb,                 steps=options["steps"], exploit=True, explore=True, title='PBT Ucb Sel')

            scores.append([score_0, score_1])
            converges.append([converge_time_0, converge_time_1])
            score_seq += [score_seq_0, score_seq_1]

            assert (pop_size is None or pop_size == pop_len0) and pop_len0 == pop_len1
            pop_size = pop_len0

            # The t score is a ratio between the difference between two groups and the difference within the groups.
            # The larger the t score, the more difference there is between groups
            # A p-value is the probability that the results from your sample data occurred by chance
            # p < 0.05 is normally accepted as valid results
            s, c = np.array(scores), np.array(converges)
            s_t, s_p = scipy.stats.ttest_ind(*s.T, equal_var=False)
            c_t, c_p = scipy.stats.ttest_ind(*c.T, equal_var=False)

            itr.set_description(f's={s.mean(axis=0).round(4)} [t,p]={np.around([s_t, s_p], 2)} | c={c.mean(axis=0).round(2)} [t,p]={np.around([c_t, c_p], 2)}')

    scores, converges, score_seq = np.array(scores), np.array(converges), np.array(score_seq)
    fig.show()

    fig, ((ax,),) = make_subplots(1, 1)
    ax.plot(score_seq[0], label='PBT Trunc Sel')
    ax.plot(score_seq[1], label='PBT Ucb Sel')
    ax.legend()
    # ax.set(title=f'Trunc vs Ucb: {dict(n=pop_size, r=options["steps_till_ready"])}', xlabel='Steps', ylabel='Ave Max Score')

    fig.show()


def run_test(make_exploiter, options=None, show=True, test_scores=None, test_converges=None):
    options = {**{
        'steps': 50, 'n': 10, 'steps_till_ready': 2, 'exploration_scale': 0.1,
        'exploit': True, 'explore': True, 'repeats': 100,
    }, **(options or {})}

    # stats
    scores, converges, score_seq = [], [], np.zeros(options['steps'])
    # iterator
    itr = (tqdm(range(options['repeats'])) if show else range(options['repeats']))

    for i in itr:
        # Members
        population = Population([
            *[ToyMember(h=ToyHyperParams(np.random.rand(2) * 0.5, 0.01), theta=np.array([.9, .9])) for i in range(options['n'])],
        ], exploiter=make_exploiter(), options=options)
        # Train
        population.train(options['steps'], exploit=options['exploit'], explore=options['explore'], show_progress=False)

        # Calculates the score as the index of the first occurrence greater than 1.18
        _firsts = np.argmax(population.scores_history > 1.18, axis=1)
        _firsts[_firsts == 0] = options['steps']
        scores.append(np.max(population.scores))
        converges.append(np.min(_firsts))
        score_seq += population.scores_history.max(axis=0) * (1 / options['steps'])

        # The t score is a ratio between the difference between two groups and the difference within the groups.
        # The larger the t score, the more difference there is between groups
        # A p-value is the probability that the results from your sample data occurred by chance
        # p < 0.05 is normally accepted as valid results
        if test_scores is not None and test_converges is not None:
            s, c, test_scores, test_converges = np.array(scores), np.array(converges), np.array(test_scores), np.array(test_converges)
            s_t, s_p = scipy.stats.ttest_ind(s, test_scores, equal_var=False)
            c_t, c_p = scipy.stats.ttest_ind(c, test_converges, equal_var=False)
            if show:
                itr.set_description(f's={s.mean().round(6)} (±{util.confidence_interval(s).round(6)}) [t,p]={np.around([s_t, s_p], 4)} | c={c.mean().round(4)} (±{util.confidence_interval(c).round(4)}) [t,p]={np.around([c_t, c_p], 4)}')
        else:
            if show:
                s, c = np.array(scores), np.array(converges)
                itr.set_description(f's={s.mean().round(6)} (±{util.confidence_interval(s).round(6)}) | c={c.mean().round(4)} (±{util.confidence_interval(c).round(4)})')

    # SCORES:       Maximum score acheived by each population
    # CONVERGES:    Minimum number of steps to converge for each population
    # SCORE_SEQ:    Average score at each time step
    scores, converges, score_seq = np.array(scores), np.array(converges), np.array(score_seq)

    return scores, converges, score_seq


if __name__ == '__main__':
    grid_search_options = dict(
        select_mode=['ucb', 'ucb_sample'],  #, 'uniform'},
        reset_mode=['exploited', 'explored_or_exploited', 'explored'],
        incr_mode=['stepped', 'exploited'],
        subset_mode=['all', 'top', 'exclude_bottom'],
        normalise_mode=['population', 'subset'],
        # test options
        steps=50,
        n=20,
        repeats=5000,
        steps_till_ready=2,
    )

    test_log = []
    def append_results(id, options, results):
        test_log.append([id, options, results])

    print('TEST: TRUNCATE', grid_search_options)
    results = run_test(lambda: ExploitTruncationSelection(), grid_search_options, show=True)
    append_results(0, grid_search_options, results)

    for i, options in enumerate(util.grid_search_named(grid_search_options)):
        def make_exploiter():
            return ExploitUcb(**{k: options[k] for k in ['subset_mode', 'incr_mode', 'reset_mode', 'select_mode', 'normalise_mode']})
        print(f'TEST UCB: {i}', options)
        r = run_test(make_exploiter, options, show=True, test_scores=results[0], test_converges=results[1])
        append_results(0, options, r)

    print('\nDONE!\n')
    save_path = f'../../../results/results_toy_{uuid4()}.pickle'
    print(f'SAVING RESULTS TO: {save_path}')
    with open(save_path, 'wb') as file:
        pickle.dump(test_log, file)


