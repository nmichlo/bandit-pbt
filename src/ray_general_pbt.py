# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import math
import random

from ray.tune.schedulers import TrialScheduler, PopulationBasedTraining
from ray.tune.schedulers.pbt import make_experiment_tag, explore
from ray.tune.trial import Trial, Checkpoint

logger = logging.getLogger(__name__)


class PopulationBasedTrainingGeneral(PopulationBasedTraining):

    def on_trial_result(self, trial_runner, trial, result):
        time = result[self._time_attr]
        state = self._trial_state[trial]

        if time - state.last_perturbation_time < self._perturbation_interval:
            return TrialScheduler.CONTINUE  # avoid checkpoint overhead

        score = self._metric_op * result[self._metric]
        state.last_score = score
        state.last_perturbation_time = time
        lower_quantile, upper_quantile = self._quantiles()

        # ↓↓↓ EXPLOIT ↓↓↓

        if trial in upper_quantile:
            state.last_checkpoint = trial_runner.trial_executor.save(trial, Checkpoint.MEMORY)
            self._num_checkpoints += 1
        else:
            state.last_checkpoint = None  # not a top trial

        if trial in lower_quantile:
            trial_to_clone = random.choice(upper_quantile)
            assert trial is not trial_to_clone
            self._exploit(trial_runner.trial_executor, trial, trial_to_clone)

        # ↑↑↑ EXPLOIT ↑↑↑

        for trial in trial_runner.get_trials():
            if trial.status in [Trial.PENDING, Trial.PAUSED]:
                return TrialScheduler.PAUSE  # yield time to other trials

        return TrialScheduler.CONTINUE


    def _quantiles(self):
        """
        Returns trials in the lower and upper `quantile` of the population.
        If there is not enough data to compute this, returns empty lists.
        """

        trials = []
        for trial, state in self._trial_state.items():
            if state.last_score is not None and not trial.is_finished():
                trials.append(trial)
        trials.sort(key=lambda t: self._trial_state[t].last_score)

        if len(trials) <= 1:
            return [], []
        else:
            num_trials_in_quantile = int(math.ceil(len(trials) * self._quantile_fraction))
            if num_trials_in_quantile > len(trials) / 2:
                num_trials_in_quantile = int(math.floor(len(trials) / 2))
            return trials[:num_trials_in_quantile], trials[-num_trials_in_quantile:]


    def _exploit(self, trial_executor, trial, trial_to_clone):
        """
        transfers perturbed state from trial_to_clone -> trial.
        If specified, also logs the updated hyperparam state.
        """

        trial_state = self._trial_state[trial]
        new_state = self._trial_state[trial_to_clone]
        if not new_state.last_checkpoint:
            logger.info("[pbt]: no checkpoint for trial. Skip exploit for Trial {}".format(trial))
            return
        new_config = explore(trial_to_clone.config, self._hyperparam_mutations, self._resample_probability, self._custom_explore_fn)
        logger.info("[exploit] transferring weights from trial {} (score {}) -> {} (score {})".format(trial_to_clone, new_state.last_score, trial, trial_state.last_score))

        if self._log_config:
            self._log_config_on_step(trial_state, new_state, trial, trial_to_clone, new_config)

        new_tag = make_experiment_tag(trial_state.orig_tag, new_config, self._hyperparam_mutations)
        reset_successful = trial_executor.reset_trial(trial, new_config, new_tag)
        if reset_successful:
            trial_executor.restore(trial, Checkpoint.from_object(new_state.last_checkpoint))
        else:
            trial_executor.stop_trial(trial, stop_logger=False)
            trial.config = new_config
            trial.experiment_tag = new_tag
            trial_executor.start_trial(trial, Checkpoint.from_object(new_state.last_checkpoint))

        self._num_perturbations += 1
        # Transfer over the last perturbation time as well
        trial_state.last_perturbation_time = new_state.last_perturbation_time

    def reset_stats(self):
        self._num_perturbations = 0
        self._num_checkpoints = 0


    def debug_string(self):
        return "PopulationBasedTraining-Improved: {} checkpoints, {} perturbs".format(self._num_checkpoints, self._num_perturbations)
