
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


from collections import defaultdict
from typing import Optional, List

from pbt.pbt import Exploiter, IPopulation, IMember
import random
import numpy as np


# ========================================================================= #
# STRATEGIES FROM: Population Based Training of Neural Networks             #
# ========================================================================= #


class ExploitTtestSelection(Exploiter):
    """
    From: Population Based Training of Neural Networks
    Section: 4.1.1 PBT for RL
        T-test selection where we uniformly sample another agent in the population,
        and compare the last 10 episodic rewards using Welch’s t-test (Welch, 1947).
        If the sampled agent has a higher mean episodic reward and satisfies the
        t-test, the weights and hyper-parameters are copied to replace the current
        agent.
    """

    def exploit(self, population: 'IPopulation', member: 'IMember') -> Optional['IMember']:
        # TODO
        raise NotImplementedError()


class ExploitTruncationSelection(Exploiter):
    """
    From: Population Based Training of Neural Networks
    Section: 4.1.1 PBT for RL
        Truncation selection where we rank all agents in the population by episodic
        reward. If the current agent is in the bottom 20% of the population, we
        sample another agent uniformly from the top 20% of the population, and copy
        its weights and hyper-parameters.
    """

    def __init__(self, bottom_ratio=0.2, top_ratio=0.2):
        assert 0 <= bottom_ratio <= 1
        assert 0 <= top_ratio <= 1
        self._bottom_ratio = bottom_ratio
        self._top_ratio = top_ratio

    def exploit(self, population: 'IPopulation', member: 'IMember') -> Optional['IMember']:
        idx_low = int(len(population) * self._bottom_ratio)
        idx_hgh = int(len(population) * (1 - self._top_ratio))

        # we rank all agents in the population by episodic reward (low to high)
        members = sorted(population, key=lambda m: m is not member)  # make sure the current member recieves the lowers priority if values are the same
        members = sorted(members, key=lambda m: m.score)

        #  If the current agent is in the bottom 20% of the population
        idx = members.index(member)
        if idx < idx_low:
            # (we choose another agent)
            return self._choose_replacement(members[:idx_low], members[idx_low:idx_hgh], members[idx_hgh:], members, population, member)

        # Otherwise we do not exploit
        return None

    def _choose_replacement(self, mbrs_low: List['IMember'], mbrs_mid: List['IMember'], mbrs_top: List['IMember'], mbrs: List['IMember'], population: 'IPopulation', member: 'IMember') -> 'IMember':
        # we sample another agent uniformly from the top 20% of the population
        return random.choice(mbrs_top)


class ExploitBinaryTournament(Exploiter):
    """
    From: Population Based Training of Neural Networks
    Section: 4.3.1 PBT for GANs
        Binary tournament in which each member of the population randomly selects
        another member of the population, and copies its parameters if the other
        member’s score is better. Whenever one member of the population is copied
        to another, all parameters (the hyperparameters, weights of the generator,
        and weights of the discriminator) are copied.
    """

    def exploit(self, population: 'IPopulation', member: 'IMember') -> Optional['IMember']:
        # TODO
        raise NotImplementedError()


# ========================================================================= #
# CUSTOM STRATEGIES                                                         #
# ========================================================================= #


class ExploitUcb(ExploitTruncationSelection):
    def __init__(self, bottom_ratio=0.2, c=0.1, mode='sample'):
        super().__init__(bottom_ratio=bottom_ratio, top_ratio=0.2)
        # UCB
        self._step_counts = defaultdict(int)
        self._c = c
        self._mode = mode
        # asdf
        assert mode in {'ordered', 'truncate', 'sample'}

    def _member_on_step(self, member): self._step_counts[member] += 1
    def _member_on_explored(self, member): self._step_counts[member] = 0
    def _member_on_exploited(self, member): pass  # self._step_counts[member] = 0

    def _choose_replacement(self, mbrs_low: List['IMember'], mbrs_mid: List['IMember'], mbrs_top: List['IMember'], mbrs: List['IMember'], population: 'IPopulation', member: 'IMember') -> 'IMember':
        assert mbrs_low + mbrs_mid + mbrs_top == mbrs
        # TODO: fix floating point for calculating top and bottom indices
        members = mbrs_top

        # assert len(_pop) == len(members), 'Not all members were included'
        # normalise scores
        scores = np.array([m.score for m in members])
        s_min, s_max = np.min(population.scores), np.max(population.scores)
        scores = (scores - s_min) / (s_max - s_min + np.finfo(float).eps)
        # normalise steps
        steps = np.array([self._step_counts[m] for m in members])
        total_steps = np.sum(steps)
        # ucb scores
        ucb_scores = ExploitUcb.ucb1(scores, steps + 1, total_steps + 1, C=self._c)
        # mode
        if self._mode == 'ordered':
            ucb_ordering = np.argsort(ucb_scores)
            return members[ucb_ordering[0]]
        elif self._mode == 'sample':
            ucb_sum = np.sum(ucb_scores)
            if ucb_sum == 0:
                ucb_scores[:] = 1
            ucb_prob = ucb_scores / (np.sum(ucb_scores) + np.finfo(float).eps)
            return members[np.random.choice(np.arange(len(members)), p=ucb_prob)]
        else:
            raise KeyError('Invalid mode')

    @staticmethod
    def ucb1(X_i, n_i, n, C=1):
        return X_i + C * np.sqrt(np.log2(n) / n_i)

# ========================================================================= \#
# END                                                                       \#
# ========================================================================= \#
