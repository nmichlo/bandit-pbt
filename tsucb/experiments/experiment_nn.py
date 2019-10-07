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
from tsucb.pbt.examples.pbt_paper_toy_example import ToyMember, ToyHyperParams
from tsucb.pbt.pbt import Population


# ========================================================================== #
# RUN EXPERIMENT                                                             #
# ========================================================================== #


def run_experiment(args: ExperimentArgs):
    EXP = comet_ml.Experiment(
        disabled=args.disable_comet,
        display_summary=True,
        parse_args=False,
    )

    EXP.add_tags([
        'toy-example',
        f'exploit-{args.pbt_exploiter}',
        'pbt'
    ])

    # OPTIONS
    print_separator(args.to_dict(useful_only=True))
    EXP.log_parameters(args.to_dict(useful_only=True))

    @util.min_time_elapsed(1.0)
    def log_step(i, score, converge_time):
        EXP.log_metrics({
            f'max_score': score,
            f'converge_time': converge_time,
        })
        tqdm.write(f'[EXPERIMENT {i+1:05d}] score: {score:8f} converge_time: {converge_time:8f}')

    # LOOP VARS
    scores, converge_times, avg_scores_per_step = [], [], np.zeros(args.pbt_steps)

    # LOOP
    for i in tqdm(range(args.experiment_repeats), 'experiment', mininterval=1, disable=os.environ.get("DISABLE_TQDM", False)):
        EXP.set_step(i)

        # MEMBERS
        population = Population([
            *[ToyMember(h=ToyHyperParams(np.random.rand(2) * 0.5, 0.01), theta=np.array([.9, .9])) for i in range(args.pbt_members)],
        ], exploiter=args.make_exploiter(), options={})
        # TRAIN
        population.train(args.pbt_steps, exploit=args.pbt_do_exploit, explore=args.pbt_do_explore, show_progress=False)

        # Calculates the score as the index of the first occurrence greater than 1.18
        _firsts = np.argmax(population.scores_history > 1.18, axis=1)
        _firsts[_firsts == 0] = args.pbt_steps
        # STEP SCORES
        score = np.max(population.scores)
        converge_time = np.min(_firsts)
        # AVERAGE SCORES
        scores.append(score)
        converge_times.append(converge_time)
        avg_scores_per_step += population.scores_history.max(axis=0) * (1 / args.experiment_repeats)

        # LOG STEP
        log_step(i, score, converge_time)

    # SCORES:       Maximum score acheived by each population
    # CONVERGES:    Minimum number of steps to converge for each population
    # SCORE_SEQ:    Average score at each time step
    scores, converge_times, avg_scores_per_step = np.array(scores), np.array(converge_times), np.array(avg_scores_per_step)

    # LOG
    ave_score, ave_scores_conf = np.average(scores), util.confidence_interval(scores, confidence=0.95)
    ave_conv_t, ave_conv_t_conf = np.average(converge_times), util.confidence_interval(converge_times, confidence=0.95)
    EXP.log_metrics({
        'ave_max_score': ave_score,     'ave_max_score_confidence_95': ave_scores_conf,
        'ave_converge_time': ave_conv_t, 'ave_converge_time_confidence_95': ave_conv_t_conf,
    })
    tqdm.write(f'[RESULT] ave_max_score:     {ave_score:8f} (±{ave_scores_conf:8f})')
    tqdm.write(f'[RESULT] ave_converge_time: {ave_conv_t:8f} (±{ave_conv_t_conf:8f})\n')

    # LOG - PLOT
    fig, ax = plt.subplots(1, 1)
    ax.plot(avg_scores_per_step)
    EXP.log_figure(f'Average Score Per Step (repeats={args.experiment_repeats})', fig)

    # END EXPERIMENT
    EXP.end()


if __name__ == '__main__':
    args = ExperimentArgs.from_system()
    run_experiment(args)

