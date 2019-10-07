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


if __name__ == '__main__':
    from tsucb.helper.util import load_dotenv
    load_dotenv()

import os
import comet_ml
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from tsucb.experiments.args_nn import ExperimentArgs
from tsucb.helper.util import print_separator
from tsucb.helper import util
from tsucb.pbt.pbt import Population


# ========================================================================== #
# RUN EXPERIMENT                                                             #
# ========================================================================== #


class ExperimentTracker(object):

    def __init__(self):
        self.COMET = None
        # LOOP VARS
        self.scores = None
        self.converge_times = None
        self.avg_scores_per_step = None

    @util.min_time_elapsed(1.0)
    def log_step(self, i, score, converge_time):
        self.COMET.log_metrics({
            f'max_score': score,
            f'converge_time': converge_time,
        })
        tqdm.write(f'[EXPERIMENT {i+1:05d}] score: {score:8f} converge_time: {converge_time:8f}')

    def pre_exp(self, exp: ExperimentArgs):
        self.COMET = comet_ml.Experiment(
            disabled=not exp.enable_comet,
            display_summary=True,
            parse_args=False,
        )

        # DETAILS
        self.COMET.add_tags([
            exp.experiment_type,
            exp.pbt_exploit_strategy,
            exp.pbt_exploit_suggest,
            'pbt'
        ])
        used_opts = exp.to_dict(used_only=True)
        print_separator(used_opts)
        self.COMET.log_parameters(used_opts)

        # AVERAGE VARS
        self.scores = []
        self.converge_times = []
        self.avg_scores_per_step = np.zeros(exp.pbt_target_steps)

    def pre_train(self, exp: ExperimentArgs, i: int):
        self.COMET.set_step(i)

    def post_train(self, exp: ExperimentArgs, population: Population):
        # Calculates the score as the index of the first occurrence greater than 1.18
        _firsts = np.argmax(population.scores_history > 1.18, axis=1)
        _firsts[_firsts == 0] = exp.tracker_converge_score

        # STEP SCORES
        score = np.max(population.scores)
        converge_time = np.min(_firsts)
        # AVERAGE SCORES
        self.scores.append(score)
        self.converge_times.append(converge_time)
        self.avg_scores_per_step += population.scores_history.max(axis=0) * (1 / args.experiment_repeats)

        # LOG STEP
        self.log_step(i, score, converge_time)

    def post_exp(self, exp: ExperimentArgs):
        # SCORES:       Maximum score acheived by each population
        # CONVERGES:    Minimum number of steps to converge for each population
        # SCORE_SEQ:    Average score at each time step
        self.scores              = np.array(self.scores)
        self.converge_times      = np.array(self.converge_times)
        self.avg_scores_per_step = np.array(self.avg_scores_per_step)

        # LOG
        ave_score       = np.average(self.scores)
        ave_conv_t      = np.average(self.converge_times)
        ave_scores_conf = util.confidence_interval(self.scores, confidence=0.95)
        ave_conv_t_conf = util.confidence_interval(self.converge_times, confidence=0.95)

        self.COMET.log_metrics({
            'ave_max_score': ave_score, 'ave_max_score_confidence_95': ave_scores_conf,
            'ave_converge_time': ave_conv_t, 'ave_converge_time_confidence_95': ave_conv_t_conf,
        })
        tqdm.write(f'[RESULT] ave_max_score:     {ave_score:8f} (±{ave_scores_conf:8f})')
        tqdm.write(f'[RESULT] ave_converge_time: {ave_conv_t:8f} (±{ave_conv_t_conf:8f})\n')

        # LOG - PLOT
        fig, ax = plt.subplots(1, 1)
        ax.plot(self.avg_scores_per_step)
        self.COMET.log_figure(f'Average Score Per Step (repeats={exp.experiment_repeats})', fig)

        # END EXPERIMENT
        self.COMET.end()


if __name__ == '__main__':
    experiment = ExperimentArgs.from_system_args()
    # tracker = ExperimentTracker()
    #
    # experiment.do_experiment(
    #     cb_pre_exp=tracker.pre_exp,
    #     cb_pre_train=tracker.pre_train,
    #     cb_post_train=tracker.post_train,
    #     cb_post_exp=tracker.post_exp,
    # )

