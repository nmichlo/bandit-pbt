import os
import time
from typing import NamedTuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import seaborn as sns
import pandas as pd

from tsucb.helper import util, defaults
from tsucb.pbt.examples.pbt_local_mnist_example import uniform_perturb, perturb, normal_explore
from tsucb.pbt.strategies import *
from tsucb.pbt.pbt import Member, Population

# SEABORN + PANDAS
defaults.set_defaults()


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
    member_options provided:
    - steps_till_ready=4
    - learning_rate=0.01
    - exploration_scale=0.1
    """

    def _setup(self, h, theta):
        self._theta = theta
        self._h = h

    def _save_theta(self):
        _THETA_STORE[self.id] = np.copy(self._theta)
    def _load_theta(self, id):
        self._theta = np.copy(_THETA_STORE[id])

    def copy_h(self) -> ToyHyperParams:
        return ToyHyperParams(np.copy(self._h.coef), self._h.alpha)
    def _set_h(self, h):
        self._h = h
    def _explored_h(self, population: 'Population') -> ToyHyperParams:
        """perturb hyper-parameters with noise from a normal distribution"""
        s = population.member_options.get('exploration_scale', 0.1)

        return ToyHyperParams(
            normal_explore(self._h.coef, s, 0, 1),
            normal_explore(self._h.alpha, s, 0, 1),
            # uniform_perturb(self._h.coef, (s-1), 1/(s-1), 0, 1),
            # uniform_perturb(self._h.alpha, (s-1), 1/(s-1), 0, 1),
            # np.clip(np.random.normal(self._h.coef, s), 0, 1),
            # np.clip(np.random.normal(self._h.alpha, s), 0, 1),
            # self._h.coef + np.random.randn(*self._h.coef.shape) * s,
            # abs(self._h.alpha + np.random.randn() * s),
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
        vals = [step.p for step in member.history]
        ax.plot(vals, color=color, lw=0.7)
    ax.axhline(y=1.2, linestyle='dotted', color='k')
    ax.set(xlim=[0, steps-1], ylim=[-0.5, 1.31], title=title, xlabel='Step', ylabel='Q')

def plot_theta(ax, population, steps, title):
    for member, color in zip(population, 'brgcmyk'):
        steps = [step for step in member.history if step.result]
        x, y = np.array([step.result['theta'][0] for step in steps]), np.array([step.result['theta'][1] for step in steps])
        jumps = np.where([step.exploit_id is not None for step in steps])[0] + 1
        x, y = np.insert(x, jumps, np.nan), np.insert(y, jumps, np.nan)
        ax.plot(x, y, color=color, lw=0.5, zorder=1)
        ax.scatter(x, y, color=color, s=1, zorder=2)

    ax.set(xlim=[-0.1, 1], ylim=[-0.1, 1], title=title, xlabel='theta0', ylabel='theta1')

def make_plot(ax_col, options, exploiter, steps=200, title=None, converge_score=1.18):
    population = Population([
        # ToyMember(ToyHyperParams(np.array([1., .0]), 0.01), np.array([.9, .9])),
        # ToyMember(ToyHyperParams(np.array([.0, 1.]), 0.01), np.array([.9, .9])),
        *[ToyMember(h=ToyHyperParams(np.random.rand(2) * 0.5, 0.01), theta=np.array([.9, .9])) for i in range(options['population_size'])],
        # *[ToyMember(ToyHyperParams(np.array([np.random.rand()*0.5, 1.]), 0.01), np.array([.9, .9])) for i in range(3)],
    ], exploiter=exploiter, member_options=options)

    t0 = time.time()
    population.train(
        steps,
        exploit=True, explore=True,
        show_progress=False,
        randomize_order=True,
        print_scores=options['print_scores']
    )
    t1 = time.time()

    # Calculates the score as the index of the first occurrence greater than 1.18
    scores = np.array([[h.p for h in m.history] for m in population])
    firsts = np.argmax(scores >= converge_score, axis=1)
    firsts[firsts == 0] = scores.shape[1]
    time_to_converge = np.min(firsts)

    score = np.max(population.scores)

    if options['repeats'] < 10:
        plot_theta(ax_col[0], population, steps=steps, title=title)
        plot_performance(ax_col[1], population, steps=steps, title=title)

    return {
        'score': score,
        'converge_time': time_to_converge,
        # scores
        'scores': scores,
        # info
        'converge_score': converge_score,
        'population_steps': steps,
        'population_size': len(population),
        'runtime': t1 - t0
    }


# ========================================================================== #
# SAME AS PBT PAPER                                                          #
# ========================================================================== #


def run_dual_test():
    options = {
        "steps": 10,
        "steps_till_ready": 2,

        "debug": False,
        "warn_exploit_self": True,
        "exploit_copies_h": False,  # must be False for toy example to be valid

        # custom
        "repeats": 500,
        "exploration_scale": 0.25,
        "population_size": 50,
        "print_scores": False,

        # redo
        'LOAD': True,
        'SAVE': False,
        'SAVE_GRAPHS': True,
    }

    make_exploit_strategy = lambda: ExploitStrategyTruncationSelection()

    # EXPLOITERS
    info_keys = ['Exploiter', 'Suggest', 'Epsilon', 'Random']
    exploiters_default = [
        (dict(Exploiter='TS Random',               Suggest='uniform', Epsilon=False, Random=True),  lambda: GeneralisedExploiter(make_exploit_strategy(), SuggestUniformRandom())),
        (dict(Exploiter='TS Greedy',               Suggest='greedy',  Epsilon=True,  Random=True),  lambda: GeneralisedExploiter(make_exploit_strategy(), SuggestGreedy())),
    ]
    exploiters_greedy = [
        (dict(Exploiter='TS ε-Greedy (ε=0.2)',   Suggest='uniform', Epsilon=True,  Random=True),  lambda: GeneralisedExploiter(make_exploit_strategy(), SuggestEpsilonGreedy(epsilon=0.2))),
        (dict(Exploiter='TS ε-Greedy (ε=0.4)',   Suggest='uniform', Epsilon=True,  Random=True),  lambda: GeneralisedExploiter(make_exploit_strategy(), SuggestEpsilonGreedy(epsilon=0.4))),
        (dict(Exploiter='TS ε-Greedy (ε=0.6)',   Suggest='uniform', Epsilon=True,  Random=True),  lambda: GeneralisedExploiter(make_exploit_strategy(), SuggestEpsilonGreedy(epsilon=0.6))),
        (dict(Exploiter='TS ε-Greedy (ε=0.8)',   Suggest='uniform', Epsilon=True,  Random=True),  lambda: GeneralisedExploiter(make_exploit_strategy(), SuggestEpsilonGreedy(epsilon=0.8))),
    ]
    exploiters_softmax = [
        (dict(Exploiter='TS Softmax (τ=0.1)',    Suggest='softmax', Epsilon=False, Random=True),  lambda: GeneralisedExploiter(make_exploit_strategy(), SuggestSoftmax(temperature=0.1))),
        (dict(Exploiter='TS Softmax (τ=0.5)',    Suggest='softmax', Epsilon=False, Random=True),  lambda: GeneralisedExploiter(make_exploit_strategy(), SuggestSoftmax(temperature=0.5))),
        (dict(Exploiter='TS Softmax (τ=1.0)',    Suggest='softmax', Epsilon=False, Random=True),  lambda: GeneralisedExploiter(make_exploit_strategy(), SuggestSoftmax(temperature=1.0))),
        (dict(Exploiter='TS Softmax (τ=10.0)',    Suggest='softmax', Epsilon=False, Random=True),  lambda: GeneralisedExploiter(make_exploit_strategy(), SuggestSoftmax(temperature=10.0))),
    ]
    exploiters_ucb = [
        (dict(Exploiter='TS UCB (c=0.1)',     Suggest='ucb',     Epsilon=False, Random=False), lambda: GeneralisedExploiter(make_exploit_strategy(), SuggestUcb(c=0.1, incr_mode='exploited'))),
        (dict(Exploiter='TS UCB (c=0.5)',     Suggest='ucb',     Epsilon=False, Random=False), lambda: GeneralisedExploiter(make_exploit_strategy(), SuggestUcb(c=0.5, incr_mode='exploited'))),
        (dict(Exploiter='TS UCB (c=1.0)',     Suggest='ucb',     Epsilon=False, Random=False), lambda: GeneralisedExploiter(make_exploit_strategy(), SuggestUcb(c=1.0, incr_mode='exploited'))),
        (dict(Exploiter='TS UCB (c=2.0)',     Suggest='ucb',     Epsilon=False, Random=False), lambda: GeneralisedExploiter(make_exploit_strategy(), SuggestUcb(c=2.0, incr_mode='exploited'))),
        # (dict(Exploiter='TS UCB (c=100, S)',     Suggest='ucb',     Epsilon=False, Random=False), lambda: OrigExploitUcb(c=0.00001, normalise_mode='none')),
        # (dict(Exploiter='TS UCB (c=100, P)',     Suggest='ucb',     Epsilon=False, Random=False), lambda: OrigExploitUcb(c=1, normalise_mode='population')),
        # (dict(Exploiter='TS UCB (c=100, C)',     Suggest='ucb',     Epsilon=False, Random=False), lambda: OrigExploitUcb(c=1, normalise_mode='none')),
    ]

    exploiters = exploiters_default + exploiters_greedy + exploiters_softmax + exploiters_ucb
    k = len(exploiters)

    if options['LOAD'] and os.path.exists('./toy_results.dat'):
        results = pd.read_msgpack('./toy_results.dat')
    else:
        fig, axs = make_subplots(2, k)
        results = []
        # EXPERIMENTS
        with tqdm(range(options['repeats']), disable=os.environ.get("DISABLE_TQDM", False)) as itr:
            for i in itr:
                for j, (info, make) in enumerate(exploiters):
                    result = make_plot(axs[:, j], options, exploiter=make(),  steps=options["steps"], title=f'PBT {info[info_keys[0]]}')
                    result.update(info)
                    results.append(result)
        results = pd.DataFrame(results)
        if options['SAVE']:
            results.to_msgpack('./toy_results.dat')

    # GATHER RESULTS - convert rows of lists to rows
    aggregated = []

    for info, group in results.groupby(info_keys):
        scores = np.array(list(group['scores']))
        max_runs = scores.max(axis=1).mean(axis=0)
        min_runs = scores.min(axis=1).mean(axis=0)
        ave_runs = scores.mean(axis=1).mean(axis=0)
        for step, (member_max, member_mean, member_min) in enumerate(zip(max_runs.T, ave_runs.T, min_runs.T)):
            # for member_max in member_max: # generate confidence bounds instead
                aggregated.append({**{k:v for k,v in zip(info_keys, info)}, 'Step': step,  'Aggregate': 'Max', 'Score': member_max})
    aggregated = pd.DataFrame(aggregated)

    # plot for old code
    plt.cla()
    plt.close()
    # plt.show()

    palette = {}
    for xpltrs, color in [(exploiters_default, 'black'), (exploiters_greedy, '#F79337'), (exploiters_softmax, '#38AFEE'), (exploiters_ucb, '#31C385')]:
        palette.update({
            e[0]['Exploiter']: v
            for e, v in zip(xpltrs, sns.light_palette(color, len(xpltrs)+1)[1:])
        })

    for i, (name, xpltrs) in enumerate([
        ('e-greedy', exploiters_default + exploiters_greedy),
        ('softmax', exploiters_default + exploiters_softmax),
        ('ucb', exploiters_default + exploiters_ucb),
    ]):
        keys = {e[0]['Exploiter'] for e in xpltrs}
        data = aggregated.loc[aggregated['Exploiter'].isin(keys)]
        # PLOT
        fig, ax = plt.subplots(figsize=(6*7/6, 2.5*7/6))
        ax.xaxis.set_major_locator(plticker.MultipleLocator(base=1.0))
        plt.xlim((2, 6))
        plt.ylim((0.2, 1.2))
        sns.lineplot(x="Step", y="Score", data=data, hue="Exploiter", palette=palette)
        if options['SAVE_GRAPHS']:
            plt.savefig(f'toy_results_{name}.png', dpi=400, bbox_inches="tight")
        plt.show()

    for i, (name, xpltrs) in enumerate([
        ('all', exploiters)
    ]):
        keys = {e[0]['Exploiter'] for e in xpltrs}
        data = aggregated.loc[aggregated['Exploiter'].isin(keys)]
        # PLOT
        fig, ax = plt.subplots(figsize=(6*7/6, 4*7/6))
        ax.xaxis.set_major_locator(plticker.MultipleLocator(base=1.0))
        plt.xlim((2, 6))
        plt.ylim((0.2, 1.2))
        sns.lineplot(x="Step", y="Score", data=data, hue="Exploiter", palette=palette)
        if options['SAVE_GRAPHS']:
            plt.savefig(f'toy_results_{name}.png', dpi=400, bbox_inches="tight")
        plt.show()



if __name__ == '__main__':
    run_dual_test()


# ========================================================================== #
# END                                                                        #
# ========================================================================== #
