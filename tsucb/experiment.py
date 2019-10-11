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
import comet_ml
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
        self.COMET: comet_ml.Experiment = None
        # LOOP VARS
        self._results: list = None

    # just in case we run this on the toy example... wait 60 seconds between calls
    @util.min_time_elapsed(60.0, wait_for_first=False)
    def log_step(self, i, result):
        s, c = result['max_score'], result['converge_time']

        self.COMET.log_metrics({
            'max_score': s,
            'converge_time': c,
        }, epoch=i, step=None)

        # CAREFUL, THIS IS REALLY EXPENSIVE IN TERMS OF THE COMET.ML API
        # CAREFUL, THIS IS REALLY EXPENSIVE IN TERMS OF THE COMET.ML API
        # CAREFUL, THIS IS REALLY EXPENSIVE IN TERMS OF THE COMET.ML API
        maxes_per_step = result['scores'].max(axis=0)
        for j, step_max_score in enumerate(maxes_per_step):
            self.COMET.log_metric('step_max_score', step_max_score, epoch=i, step=j)

        tqdm.write(f'[EXPERIMENT {i+1:05d}] max_score={s:8f} converge_time={c:8f}')

    def pre_exp(self, exp: ExperimentArgs):
        self.COMET = comet_ml.Experiment(
            project_name=exp.comet_project_name,
            disabled=not exp.comet_enable,
            display_summary=True,
            # auto stuffs
            log_graph=False,  # Computation Graph
            auto_metric_logging=False,  # Loss, etc.
            parse_args=False,  # command line args
        )

        self.COMET.set_name(exp.experiment_name)

        # LOG
        exp.print_reproduce_info()
        exp.print_args()
        exp.print_dict_computed()

        # DETAILS
        self.COMET.add_tags([
            exp.experiment_type,
            exp.pbt_exploit_strategy,
            exp.pbt_exploit_suggest,
            'pbt'
        ])
        # LOG: parameters
        used_opts = exp.as_dict(used_only=True)
        self.COMET.log_parameters(used_opts)

        # LOG: computed parameters
        computed_opts = exp.get_dict_computed()
        self.COMET.log_others(computed_opts)

        # INITIALISE:
        self._results = []

        # BEGIN:
        util.print_separator('RUNNING EXPERIMENT:')

    def pre_run(self, exp: ExperimentArgs, i: int):
        self.COMET.set_epoch(i)
        self.COMET.set_step(i)

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
                step_result=h.result,
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
        self.log_step(i, self._results[-1])

    def post_exp(self, exp: ExperimentArgs):
        util.print_separator('EXPERIMENT RESULTS:')

        # EXTRACT SCORES:
        extracted_scores = np.array([r['max_score'] for r in self._results])
        extracted_converge_times = np.array([r['converge_time'] for r in self._results])
        # CALCULATE AVERAGES:
        ave_max_score       = np.average(extracted_scores)
        ave_converge_time   = np.average(extracted_converge_times)
        ave_scores_conf = util.confidence_interval(extracted_scores, confidence=0.95)
        ave_conv_time_conf = util.confidence_interval(extracted_converge_times, confidence=0.95)

        # LOG:
        self.COMET.log_metrics({
            'ave_max_score': ave_max_score, 'ave_max_score_conf_95': ave_scores_conf,
            'ave_converge_time': ave_converge_time, 'ave_converge_time_conf_95': ave_conv_time_conf,
        })

        # COMPUTE:
        extracted_step_maxes = np.array([np.max(r["scores"], axis=0) for r in self._results])
        mean_step_maxes = extracted_step_maxes.mean(axis=0)

        # LOG
        tqdm.write('[MAX SCORES OVER RUNS]:')
        for i, run_maxes in enumerate(extracted_step_maxes):
            tqdm.write(f'    {i}: {run_maxes.tolist()}')
        tqdm.write(f'\n[RESULT] mean_step_maxes:   {mean_step_maxes.tolist()})')
        tqdm.write(f'                            ±{[util.confidence_interval(step_maxes, confidence=0.95) for step_maxes in extracted_step_maxes.T]})')
        tqdm.write(f'\n[RESULT] ave_max_score:     {ave_max_score:8f} (±{ave_scores_conf:8f})')
        tqdm.write(f'[RESULT] ave_converge_time: {ave_converge_time:8f} (±{ave_conv_time_conf:8f})\n')

        # try:
        #     df = pd.DataFrame({
        #         'score': mean_step_maxes,
        #         'step': np.arange(len(mean_step_maxes)),
        #     })
        #     plt.figure(figsize=(6, 3.75))
        #     sns.lineplot(x='step', y='score', data=df, palette=sns.color_palette("GnBu", 1))
        #     plt.show()
        # except Exception as e:
        #     traceback.print_exc(e)

        try:
            result_file = os.path.join(util.make_dir(exp.results_dir), f'results_{exp.start_time_str}_{exp.experiment_name}_{experiment.experiment_id}.npz')
            np.savez_compressed(result_file, dict(
                results=self._results,
                arguments=exp.as_dict(used_only=True, exclude_defaults=False)
            ))
            tqdm.write(f'[SAVED]: {result_file}\n')
        except Exception as e:
            traceback.print_exc(e)

        util.print_separator('EXPERIMENT ENDED!')

        # END EXPERIMENT
        self.COMET.end()


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
        # DO NOT CHANGE - EXPERIMENT - THESE ARE ALREADY DEFAULTS:
        experiment_type='cnn',
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