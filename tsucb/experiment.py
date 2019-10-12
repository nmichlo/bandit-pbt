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


# ========================================================================== #
# LOAD ENV                                                                   #
# ========================================================================== #


if __name__ == '__main__':
    from tsucb.helper.util import load_dotenv
    load_dotenv()


# ========================================================================== #
# IMPORTS                                                                    #
# ========================================================================== #


import os
import traceback

import wandb
import numpy as np
from tqdm import tqdm

from tsucb.experiment_args import ExperimentArgs, ExperimentTracker
from tsucb.helper import util, defaults
from tsucb.pbt.pbt import Population


# ========================================================================== #
# DEFAULTS                                                                   #
# ========================================================================== #


# SEABORN & PANDAS DEFAULTS
defaults.set_defaults()


# ========================================================================== #
# RUN EXPERIMENT                                                             #
# ========================================================================== #


class ExperimentTrackerConvergence(ExperimentTracker):

    def __init__(self):
        # LOOP VARS
        self._results: list = None

    @util.min_time_elapsed(5.0, wait_for_first=False)
    def print_step(self, i, result):
        s, c = result['max_score'], result['converge_time']
        tqdm.write(f'[EXPERIMENT {i+1:05d}] max_score={s:8f} converge_time={c:8f}')

    def pre_exp(self, exp: ExperimentArgs):
        if not exp.comet_enable:
            os.environ['WANDB_MODE'] = 'dryrun'

        wandb.init(
            name=f'{exp.experiment_name} ({exp.experiment_id[:5]})',
            group=exp.experiment_name,
            project=exp.comet_project_name,
            id=f'{exp.experiment_name}-{exp.experiment_id}',
            job_type='pbt',
            tags=[
                exp.experiment_type,
                exp.pbt_exploit_strategy,
                exp.pbt_exploit_suggest,
            ],
            config={
                **exp.as_dict(used_only=True),
                **exp.get_dict_computed(),
            },
        )
        # LOG
        exp.print_reproduce_info()
        exp.print_args()
        exp.print_dict_computed()
        # INITIALISE:
        self._results = []
        # BEGIN:
        util.print_separator('RUNNING EXPERIMENT:')

    def pre_run(self, exp: ExperimentArgs, i: int):
        pass

    def post_run(self, exp: ExperimentArgs, i: int, population: Population):
        # CONVERGE TIME: The index of the first occurrence which has a score >= exp.tracker_converge_score
        _firsts = np.argmax(population.scores_history > exp.tracker_converge_score, axis=1)
        _firsts[_firsts == 0] = exp.pbt_target_steps
        converge_time = np.min(_firsts)

        # ALL HISTORIES:
        histories = [[
            dict(
                hyper_parameters=h.h,
                score=h.p,
                step=h.t,
                exploit_id=h.exploit_id,
            ) for h in m.history
        ] for m in population.members]

        # ALL SCORES:
        scores = np.array([[h.p for h in m.history] for m in population.members])

        # APPEND RESULTS:
        self._results.append({
            'experiment_id': exp.experiment_id,
            'run': i,
            # final scores
            'converge_time': converge_time,
            'max_score': np.max(scores),
            # all data:
            'scores': scores,
            'histories': histories
        })

        # LOG STEP | COMET & PRINT
        self.print_step(i, self._results[-1])

    def post_exp(self, exp: ExperimentArgs):
        util.print_separator('EXPERIMENT RESULTS:')

        # RUN STEP SCORES
        run_scores          = np.array([r["scores"] for r in self._results])
        run_step_max_scores = run_scores.max(axis=1)
        ave_step_max_scores = run_step_max_scores.mean(axis=0)

        wandb.summary['run_scores'] = run_scores
        wandb.summary['run_step_max_scores'] = run_step_max_scores
        wandb.summary['ave_step_max_scores'] = ave_step_max_scores

        # RUN SCORES
        run_max_scores = run_step_max_scores.max(axis=1)
        avg_max_score  = run_max_scores.mean()

        wandb.summary['run_max_scores'] = run_max_scores
        wandb.summary['avg_max_score']  = avg_max_score

        # RUN CONVERGENCE
        run_converge_times      = np.array([r['converge_time'] for r in self._results])
        avg_converge_time = run_converge_times.mean()

        wandb.summary['run_converge_times'] = run_converge_times
        wandb.summary['avg_converge_time']  = avg_converge_time

        wandb.summary['run_histories'] = [r["histories"] for r in self._results]

        # WANDB: STEPS:
        for i, ave_step_max_score in enumerate(ave_step_max_scores):
            wandb.log({
                'ave_step_max_score': ave_step_max_score
            }, commit=True, step=i)

        util.print_separator('EXPERIMENT ENDED!')



if __name__ == '__main__':

    # 0.65 minutes per epoch per member
    # ie. 0.65 * 50 members * 5 epochs = 2.70 hours
    #     0.65 * 50 members * 3 epochs = 1.62 hours
    #     0.65 * 25 members * 5 epochs = 1.35 hours
    #     0.65 * 25 members * 3 epochs = 0.81 hours
    #     0.65 * 10 members * 5 epochs = 0.54 hours
    #     0.65 * 10 members * 3 epochs = 0.33 hours

    experiment = ExperimentArgs.from_system_args(defaults=dict(
        # DO NOT CHANGE - MISC
        pbt_print=False,
        comet_enable=False,
        experiment_repeats=5,
        # DO NOT CHANGE - EXPERIMENT - THESE ARE ALREADY DEFAULTS:
        experiment_type='toy',
        cnn_steps_per_epoch=5,
        pbt_target_steps=5*4,  # 5 steps per epoch for 4 epochs
        pbt_exploit_strategy='ts',
        # ALLOW CHANGES:
        experiment_name='random',
        pbt_members=25,
        pbt_exploit_suggest='ran',
    ))

    experiment.do_experiment(
        tracker=ExperimentTrackerConvergence(),
    )

    # [COMMAND MINIMAL]:
    #     $ workspace/research/improving-pbt/tsucb/experiment.py --experiment-name="random" --pbt-print --comet-enable
    # [COMMAND USED]:
    #     $ workspace/research/improving-pbt/tsucb/experiment.py --experiment-repeats="1" --experiment-name="random" --experiment-id="c59f5226-fb04-46a3-ae31-0a49e9aa555f" --experiment-type="cnn" --experiment-seed="2195761148" --cnn-dataset="MNIST" --cnn-batch-size="32" --cnn-steps-per-epoch="5" --pbt-print --pbt-target-steps="15" --pbt-members="25" --pbt-members-ready-after="2" --pbt-exploit-strategy="ts" --pbt-exploit-suggest="random" --strategy-ts-ratio-top="0.2" --strategy-ts-ratio-bottom="0.2" --comet-enable --comet-project-name="unnamed-project"
    # [COMMAND ALL]:
    #     $ workspace/research/improving-pbt/tsucb/experiment.py --experiment-repeats="1" --experiment-name="random" --experiment-id="c59f5226-fb04-46a3-ae31-0a49e9aa555f" --experiment-type="cnn" --experiment-seed="2195761148" --cnn-dataset="MNIST" --cnn-batch-size="32" --cnn-steps-per-epoch="5" --pbt-print --pbt-target-steps="15" --pbt-members="25" --pbt-members-ready-after="2" --pbt-exploit-strategy="ts" --pbt-exploit-suggest="random" --suggest-ucb-incr-mode="exploited" --suggest-ucb-c="1.0" --suggest-softmax-temp="1.0" --suggest-eps="0.75" --strategy-ts-ratio-top="0.2" --strategy-ts-ratio-bottom="0.2" --strategy-tt-confidence="0.95" --comet-enable --comet-project-name="unnamed-project"
