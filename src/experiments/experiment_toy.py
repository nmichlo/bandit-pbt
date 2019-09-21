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


import dotenv

# LOAD ENV
dotenv.load_dotenv(dotenv.find_dotenv(), verbose=True)


import os
import pickle

import comet_ml
import scipy.stats
from tqdm import tqdm
from helper import util
from uuid import uuid4

from pbt.examples.pbt_paper_toy_example import ToyMember, ToyHyperParams
from pbt.pbt import Population
from pbt.strategies import ExploitUcb, ExploitTruncationSelection
import numpy as np


# ========================================================================= #
# SYSTEM                                                                    #
# ========================================================================= #


class ToyExperimentLogger(object):
    def __init__(self, experiment, is_ucb=True):
        self.is_ucb = is_ucb
        # COMET.ML
        self._exp = experiment
        # TOY-EXAMPLE: the type of member
        # EXPLOIT-UCB: exploitation strategy
        # PBT: pbt or ray-pbt
        self._exp.add_tags(['toy-example', 'exploit-ucb' if self.is_ucb else 'exploit-truncation', 'pbt'])
        self._exp.set_step(0)

    def log_options(self, options, print_=True):
        params = {
            # EXPERIMENT
            'steps': options['steps'],
            'n': options['n'],
            'repeats': options['repeats'],
            'steps_till_ready': options['steps_till_ready'],
            # UCB
            **({
                   'ucb_select_mode': options['select_mode'],
                   'ucb_reset_mode': options['reset_mode'],
                   'ucb_incr_mode': options['incr_mode'],
                   'ucb_subset_mode': options['subset_mode'],
                   'ucb_normalise_mode': options['normalise_mode'],
                    **({'ucb_c': options['c']} if 'c' in options else {}),
               } if self.is_ucb else {})
        }
        if print_:
            print('\n#', '=' * 100, "#")
            print(params)
            print('#', '=' * 100, "#\n")
        self._exp.log_parameters(params)
        return params

    def log_averages(self, name, values, values_target=None):
        values = np.array(values)
        self._exp.log_metrics({
            f'ave_{name}': np.average(values),
            f'ave_{name}_confidence': util.confidence_interval(values),
        })
        if values_target is not None:
            values_target = np.array(values_target)
            s_t, s_p = np.nan_to_num(scipy.stats.ttest_ind(values, values_target, equal_var=False))
            self._exp.log_metrics({
                f'ave_{name}_t_value': s_t,
                f'ave_{name}_p_value': s_p,
            })

    def end(self):
        # FINISH THE EXPERIMENT
        self._exp.end()
        self._exp = None


# ========================================================================== #
# RUN EXPERIMENTS                                                            #
# ========================================================================== #


def run_experiment(make_exploiter, options=None, show=True, test_scores=None, test_converges=None, test_seq=None):
    is_ucb = isinstance(make_exploiter(), ExploitUcb)

    EXP = comet_ml.Experiment()
    EXP_HELPER = ToyExperimentLogger(EXP, is_ucb=is_ucb)

    # OPTIONS
    options = {**{
        'steps': 50, 'n': 10, 'steps_till_ready': 2, 'exploration_scale': 0.1,
        'exploit': True, 'explore': True, 'repeats': 100,
    }, **(options or {})}
    EXP_HELPER.log_options(options)
    print('OPTIONS:', options)

    # LOOP VARS
    scores, converges, score_seq = [], [], np.zeros(options['steps'])
    itr = (tqdm(range(options['repeats'])) if show else range(options['repeats']))

    # LOOP
    for i in itr:
        EXP.set_step(i+1)

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

    # LOG
    EXP_HELPER.log_averages('max_score', scores, test_scores)
    EXP_HELPER.log_averages('converge_time', converges, test_converges)

    return scores, converges, score_seq


def run_tests():
    grid_search_options = dict(
        c=[0.1, 0.05, 0.2],
        select_mode=['ucb'], #, 'ucb_sample'],  #, 'uniform'}, # UCB SAMPLE IS USELESS
        reset_mode=['exploited', 'explored_or_exploited', 'explored'],
        subset_mode=['all', 'top', 'exclude_bottom'],
        normalise_mode=['population', 'subset'],
        incr_mode=['exploited', 'stepped'],
        # test options
        steps=20,
        n=20,
        repeats=2500,
        steps_till_ready=2,
    )

    # HELPER
    test_log = []
    def append_results(id, options, results):
        test_log.append([id, options, results])

    # GRID SEARCH OPTIONS
    search_options = list(enumerate(util.grid_search_named(grid_search_options)))
    print(f'[GRID SEARCH PERMUTATIONS]: {len(search_options)}')

    # FLITER SEARCH OPTIONS
    exclude_options = []
    keys = ['select_mode', 'reset_mode', 'incr_mode', 'subset_mode', 'normalise_mode', 'c']
    exclude_options = {tuple(s[k] for k in keys) for s in exclude_options}
    search_options = [(i, s) for i, s in search_options if tuple(s[k] for k in keys) not in exclude_options]
    print(f'[GRID SEARCH REMAINING]: {len(search_options)}')

    # RUN BASELINE EXPERIMENT
    results = run_experiment(lambda: ExploitTruncationSelection(), grid_search_options, show=True)
    append_results(0, grid_search_options, results)

    # RUN EXPERIMENTS
    for i, options in search_options:
        def make_exploiter():
            return ExploitUcb(**{k: options[k] for k in ['subset_mode', 'incr_mode', 'reset_mode', 'select_mode', 'normalise_mode', 'c']})
        r = run_experiment(make_exploiter, options, show=True) #, test_scores=results[0], test_converges=results[1])
        append_results(0, options, r)

    print('\nDONE!\n')
    save_path = f'../../../results/results_toy_{uuid4()}.pickle'
    print(f'SAVING RESULTS TO: {save_path}')
    with open(save_path, 'wb') as file:
        pickle.dump(test_log, file)


if __name__ == '__main__':
    os.environ['COMET_PROJECT_NAME'] = 'improving-pbt-toy-examples-fixes-3'
    run_tests()


# ========================================================================== #
# UPLOAD OLD SAVED EXPERIMENTS                                               #
# ========================================================================== #


# def upload_experiment(
#         id, options,
#         scores, converges, score_seq,
#         trunc_scores=None, trunc_converges=None, trunc_score_seq=None, is_ucb=True
# ):
#     assert len(scores) == len(converges)
#     EXP = comet_ml.OfflineExperiment(
#         # DISABLE
#         disabled=False,
#         # CODE, ENV, GIT
#         log_code=False,
#         log_env_details=False,
#         log_git_metadata=False,
#         log_git_patch=False,
#         # HOST, GPU, CPU
#         log_env_gpu=False,
#         log_env_host=False,
#         log_graph=False,
#     )
#     logger = ToyExperimentLogger(EXP, is_ucb=is_ucb)
#     logger.log_options(options)
#     logger.log_averages('max_score', scores, trunc_scores)
#     logger.log_averages('converge_time', converges, trunc_converges)
#     logger.end()
#
#
# def upload_results_to_comet(test_log):
#     # MAKE DIRS
#     if not os.path.isdir(os.environ['COMET_OFFLINE_DIRECTORY']):
#         print('[MAKING DIRS]', os.environ['COMET_OFFLINE_DIRECTORY'])
#         os.makedirs(os.environ['COMET_OFFLINE_DIRECTORY'], exist_ok=True)
#
#     # LOAD TRUNCATION SELECTION DATA
#     trunc_id, trunc_options, (trunc_scores, trunc_converges, trunc_score_seq) = test_log[0]
#     upload_experiment(trunc_id, trunc_options, trunc_scores, trunc_converges, trunc_score_seq, is_ucb=False)
#
#     # LOAD UCB DATA
#     for id, options, (scores, converges, score_seq) in tqdm(test_log[1:]):
#         upload_experiment(id, options, scores, converges, score_seq, trunc_scores, trunc_converges, trunc_score_seq, is_ucb=True)


# if __name__ == '__main__':
#     with open('../../../results/results_toy-pbt_ucb-vs-trunv_grid-search_5000iters.pickle', 'rb') as file:
#         test_log = pickle.load(file)
#         upload_results_to_comet(test_log)



# ========================================================================= \#
# END                                                                       \#
# ========================================================================= \#
