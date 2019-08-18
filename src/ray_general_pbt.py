# coding=utf-8


import itertools
import json
import logging
import math
import os
import random
import shutil
from typing import Dict, Tuple, Iterator, List, Optional

import numpy as np
from ray.tune import TuneError
from ray.tune.ray_trial_executor import RayTrialExecutor
from ray.tune.result import TRAINING_ITERATION
from ray.tune.schedulers import TrialScheduler, FIFOScheduler
from ray.tune.schedulers.pbt import make_experiment_tag, explore
from ray.tune.trial import Trial, Checkpoint
from ray.tune.trial_runner import TrialRunner

logger = logging.getLogger(__name__)


# ========================================================================== #
# Internal PBT State                                                         #
# ========================================================================== #


class TrialState(object):
    """Internal PBT state tracked per-trial."""

    def __init__(self, trial):
        self.orig_tag = trial.experiment_tag
        self.last_score = None
        self.last_checkpoint = None
        self.last_perturbation_time = 0
        self.num_explorations = 0
        self.num_steps = 0

    def __str__(self):
        return str((self.last_score, self.last_checkpoint, self.last_perturbation_time))

    def __repr__(self):
        return str(self)


TrialsStatesDict = Dict[Trial, TrialState]


# ========================================================================== #
# Exploit Mechanisms                                                         #
# ========================================================================== #


class Exploiter(object):

    @staticmethod
    def get_active_trials(trials_states_dict: TrialsStatesDict) -> List[Trial]:
        trials = []
        for trial, state in trials_states_dict.items():
            if state.last_score is not None and not trial.is_finished():
                trials.append(trial)
        return trials

    def exploit(self, pbt: 'GeneralPbt', trial: Trial, trials_states_dict: TrialsStatesDict) -> Tuple[List[Trial], List[Trial]]:
        """
        Returns trials in the lower and upper `quantile` of the population.
        If there is not enough data to compute this, returns empty lists.
        """
        raise NotImplementedError('Override Me')

    def __str__(self):
        return "({})".format(self.__class__.__name__)
    def __repr__(self):
        return str(self)


class QuantileExploiter(Exploiter):
    """Exploit function for PBT"""

    def __init__(self, quantile_fraction=0.25):
        if quantile_fraction > 0.5 or quantile_fraction < 0:
            raise TuneError("You must set `quantile_fraction` to a value between 0 and 0.5. Current value: '{}'".format(quantile_fraction))
        self._quantile_fraction = quantile_fraction

    def exploit(self, pbt: 'GeneralPbt', trial: Trial, trials_states_dict: TrialsStatesDict) -> Tuple[List[Trial], List[Trial]]:
        """
        Returns trials in the lower and upper `quantile` of the population.
        If there is not enough data to compute this, returns empty lists.
        """
        trials = Exploiter.get_active_trials(trials_states_dict)

        if len(trials) <= 1:
            return [], []

        # calculate scores
        trials.sort(key=lambda t: trials_states_dict[t].last_score)

        # split trials
        num_trials_in_quantile = int(math.ceil(len(trials) * self._quantile_fraction))
        if num_trials_in_quantile > len(trials) / 2:
            num_trials_in_quantile = int(math.floor(len(trials) / 2))

        return trials[:num_trials_in_quantile], trials[-num_trials_in_quantile:]

    def __str__(self):
        return "({}: quantile_fraction: {})".format(self.__class__.__name__, self._quantile_fraction)


class UcbExploiter(Exploiter):
    """Exploit function for PBT"""

    def __init__(self):
        self._max_score = float('-inf')

    @staticmethod
    def ucb1(X_i, n_i, n, C=1):
        return X_i + C * np.sqrt(np.log2(n+1) / (n_i+1))

    def exploit(self, pbt: 'GeneralPbt', trial: Trial, trials_states_dict: TrialsStatesDict) -> Tuple[List[Trial], List[Trial]]:
        """
        UCB extension of exploiting
        """

        trials = Exploiter.get_active_trials(trials_states_dict)

        if len(trials) <= 1:
            return [], []

        # Update Max
        self._max_score = max(self._max_score, trials_states_dict[trial].last_score)

        # TODO: this shouldn't be average because of synchronisation
        ave_steps = np.average([trials_states_dict[t].num_steps for t in trials])

        probabilities = [UcbExploiter.ucb1(
            X_i=s.last_score / self._max_score if self._max_score != 0 else 0,
            n_i=s.num_steps,  # TODO: move out into Exploiter
            n=ave_steps,  # TODO: move out into Exploiter
            C=1
        ) for t, s in zip(trials, (trials_states_dict[t] for t in trials))]

        index = np.argmax(probabilities)
        choice = trials[index]
        del trials[index]

        # choice = np.random.choice(trials, p=probabilities/np.sum(probabilities))
        # trials.remove(choice)

        return trials, [choice]

    def __str__(self):
        return "({}: max_score: {})".format(self.__class__.__name__, self._max_score)


# ========================================================================== #
# Generalised PBT                                                            #
# ========================================================================== #


class GeneralPbt(FIFOScheduler):

    def __init__(
            self,
            exploit_metric_attr="episode_reward_mean",
            exploit_metric_mode="max",
            exploiter=QuantileExploiter(),

            explore_counter_attr="time_total_s",
            explore_counter_interval=60.0,
            explore_func=None,

            hyperparam_mutations=None,
            hyperparam_mutate_probability=0.25,

            log_config=True
    ):
        if not hyperparam_mutations and not explore_func:
            raise TuneError("You must specify at least one of `hyperparam_mutations` or `custom_explore_func` to use PBT.")
        if exploit_metric_mode not in ["min", "max"]:
            raise TuneError("`mode` must be 'min' or 'max'!")
        if not isinstance(exploiter, Exploiter):
            raise TuneError("`exploiter` must be an instance of type `Exploiter`")

        super().__init__()

        if hyperparam_mutations is None:
            hyperparam_mutations = {}

        self._exploit_metric_attr = exploit_metric_attr
        self._exploit_metric_sign = 1 if (exploit_metric_mode == "max") else -1
        self._exploiter = exploiter

        self._explore_counter_attr = explore_counter_attr
        self._explore_counter_interval = explore_counter_interval
        self._explore_func = explore_func

        self._hyperparam_mutations = hyperparam_mutations
        self._hyperparam_mutate_probability = hyperparam_mutate_probability

        self._log_config = log_config

        self._trials_states_dict: TrialsStatesDict = {}

        # Metrics
        self._num_checkpoints = 0
        self._num_explorations = 0

    def on_trial_add(self, trial_runner: TrialRunner, trial: Trial):
        self._trials_states_dict[trial] = TrialState(trial)

    def on_trial_result(self, trial_runner: TrialRunner, trial: Trial, result: dict):
        time = result[self._explore_counter_attr]
        state = self._trials_states_dict[trial]

        # CHECK HOW LONG THE TRIAL HAS RUN FOR
        if time - state.last_perturbation_time < self._explore_counter_interval:
            return TrialScheduler.CONTINUE  # avoid checkpoint overhead


        # SET LAST STATE VALUES
        state.last_score = self._exploit_metric_sign * result[self._exploit_metric_attr]  # negate if exploit_metric_mode="min"
        state.last_perturbation_time = time
        state.num_steps += 1

        # EXPLOIT FROM ALL THE OTHER TRAILS
        lower_quantile, upper_quantile = self._exploiter.exploit(self, trial, self._trials_states_dict)

        # TRAIL IS GOOD - SAVE
        if trial in upper_quantile:
            state.last_checkpoint = trial_runner.trial_executor.save(trial, Checkpoint.MEMORY)
            self._num_checkpoints += 1
        else:
            state.last_checkpoint = None  # not a top trial

        # TRAIL IS BAD - REPLACE
        if trial in lower_quantile:
            exploiting_trial = random.choice(upper_quantile)
            assert trial is not exploiting_trial
            self._exploit_trial(trial_runner.trial_executor, trial, exploiting_trial)

        # IF ANOTHER TRIAL NEEDS TO BE RUN, PAUSE THIS ONE
        for trial in trial_runner.get_trials():
            if trial.status in [Trial.PENDING, Trial.PAUSED]:
                return TrialScheduler.PAUSE  # yield time to other trials

        return TrialScheduler.CONTINUE

    def _log_config_on_step(self, trial_state: TrialState, new_state: TrialState, trial: Trial, trial_to_clone: Trial, new_config: dict):
        # new_config is dict of hyper-params

        """
        Logs transition during exploit/exploit step.
        For each step, logs: [target trial tag, clone trial tag, target trial
        iteration, clone trial iteration, old config, new config].
        """
        trial_name, trial_to_clone_name = (trial_state.orig_tag, new_state.orig_tag)
        trial_id = "".join(itertools.takewhile(str.isdigit, trial_name))
        trial_to_clone_id = "".join(itertools.takewhile(str.isdigit, trial_to_clone_name))
        trial_path = os.path.join(trial.local_dir, "pbt_policy_" + trial_id + ".txt")
        trial_to_clone_path = os.path.join(trial_to_clone.local_dir, "pbt_policy_" + trial_to_clone_id + ".txt")
        policy = [
            trial_name,
            trial_to_clone_name,
            trial.last_result.get(TRAINING_ITERATION, 0),
            trial_to_clone.last_result.get(TRAINING_ITERATION, 0),
            trial_to_clone.config,
            new_config
        ]
        # Log to global file.
        with open(os.path.join(trial.local_dir, "pbt_global.txt"), "a+") as f:
            f.write(json.dumps(policy) + "\n")
        # Overwrite state in target trial from trial_to_clone.
        if os.path.exists(trial_to_clone_path):
            shutil.copyfile(trial_to_clone_path, trial_path)
        # Log new exploit in target trial log.
        with open(trial_path, "a+") as f:
            f.write(json.dumps(policy) + "\n")

    def _exploit_trial(self, trial_executor: RayTrialExecutor, trial: Trial, trial_to_clone: Trial):
        """
        Transfers perturbed state from trial_to_clone -> trial.
        If specified, also logs the updated hyperparam state.
        """

        trial_state = self._trials_states_dict[trial]
        new_state = self._trials_states_dict[trial_to_clone]
        if not new_state.last_checkpoint:
            logger.info("[pbt]: no checkpoint for trial. Skip exploit for Trial {}".format(trial))
            return
        new_config = explore(trial_to_clone.config, self._hyperparam_mutations, self._hyperparam_mutate_probability, self._explore_func)
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

        # TODO: move to Exploiter
        new_state.num_steps = 0
        trial_state.num_steps = 0
        new_state.num_explorations = 0
        trial_state.num_explorations += 1
        self._num_explorations += 1
        # Transfer over the last perturbation time as well
        trial_state.last_perturbation_time = new_state.last_perturbation_time

    def choose_trial_to_run(self, trial_runner: TrialRunner):
        """
        Ensures all trials get fair share of time (as defined by time_attr).
        This enables the PBT scheduler to support a greater number of
        concurrent trials than can fit in the cluster at any given time.
        """

        candidates = []
        for trial in trial_runner.get_trials():
            if trial.status in [Trial.PENDING, Trial.PAUSED] and trial_runner.has_resources(trial.resources):
                candidates.append(trial)
        candidates.sort(key=lambda trial: self._trials_states_dict[trial].last_perturbation_time)
        return candidates[0] if candidates else None

    def reset_stats(self):
        self._num_explorations = 0
        self._num_checkpoints = 0

    def last_scores(self, trials):
        scores = []
        for trial in trials:
            state = self._trials_states_dict[trial]
            if state.last_score is not None and not trial.is_finished():
                scores.append(state.last_score)
        return scores

    def debug_string(self):
        return "PopulationBasedTraining: {} checkpoints, {} perturbs".format(self._num_checkpoints, self._num_explorations)

# ========================================================================== #
# End                                                                        #
# ========================================================================== #
