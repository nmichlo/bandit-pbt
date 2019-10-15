
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
from typing import Optional, List, NoReturn
from tqdm import tqdm
from tsucb.helper.util import sorted_random_ties, argsorted_random_ties
from tsucb.pbt.pbt import Exploiter, IPopulation, IMember, PopulationListener, MergedPopulationListener
import random
import numpy as np


# ========================================================================= #
# SUGGESTION STRATEGIES                                                     #
# ========================================================================= #


class ISuggest(PopulationListener):

    # SUGGEST
    def suggest(self, filtered: List['IMember']) -> IMember:
        raise NotImplementedError()

    def __str__(self):
        return self.__class__.__name__
    def __repr__(self):
        return str(self)

class _SuggestRandomGreedy(ISuggest):

    def __init__(self, wrapped_suggest, epsilon=0.5):
        """
        :param epsilon: Chance to select wrapped_suggest, otherwise makes the greedy choice.
        """
        assert 0 <= epsilon <= 1
        assert isinstance(wrapped_suggest, ISuggest)
        self._epsilon = epsilon
        self._wrapped_suggest = wrapped_suggest
        # >>> OVERRIDE LISTENERS <<<
        self._wrapped_suggest.assign_listeners_to(self)

    def suggest(self, filtered: List['IMember']) -> IMember:
        if np.random.random() < self._epsilon:
            return self._wrapped_suggest.suggest(filtered)
        else:
            return filtered[np.argmax([m.score for m in filtered])]

class SuggestGreedy(ISuggest):
    def suggest(self, filtered: List['IMember']) -> IMember:
        return filtered[np.argmax([m.score for m in filtered])]

class SuggestUniformRandom(ISuggest):
    def suggest(self, filtered: List['IMember']) -> IMember:
        return random.choice(filtered)

class SuggestSoftmax(ISuggest):
    def __init__(self, temperature=1.0, normalise=True):
        self._temperature = temperature
        self._normalise = normalise

    def suggest(self, filtered: List['IMember']) -> IMember:
        scores = np.array([m.score for m in filtered])
        # NORMALISE SCORES BETWEEN 0 AND 1
        if self._normalise:
            s_min, s_max = np.min(scores), np.max(scores)
            scores = (scores - s_min) / (s_max - s_min + np.finfo(float).eps)
        # AVOID OVERFLOW:
        scores = scores - np.max(scores)  # this should not change the values
        # CALCULATE PROB
        vals = np.exp(scores / self._temperature)
        prob = vals / np.sum(vals)
        # RETURN CATEGORICALLY SAMPLED
        return np.random.choice(filtered, p=prob)

class SuggestUcb(ISuggest):
    def __init__(self, c=1.0, incr_mode='exploited', debug=False):
        # UCB
        self.__step_counts = defaultdict(int)
        self._c = c
        # MODES
        assert incr_mode in {'stepped', 'exploited'}
        self._incr_mode = incr_mode
        # debug
        self.debug = debug

    def _visits_incr(self, member):
        assert isinstance(member, IMember)
        self.__step_counts[member] += 1
    def _visits_reset(self, member):
        assert isinstance(member, IMember)
        self.__step_counts[member] = 0
    def _visits_get(self, member) -> int:
        assert isinstance(member, IMember)
        return self.__step_counts[member]

    def _debug_message(self, message, member=None):
        if self.debug:
            extra = list(self.__step_counts.keys()).index(member) if member and member in self.__step_counts else ''
            tqdm.write(f'{message:>10s}: {list(self.__step_counts.values())} {extra}')

    # THESE TWO STRATEGIES ARE EFFECTIVELY THE SAME - ALWAYS HAPPEN TOGETHER
    def _member_on_explored(self, member):
        self._debug_message('EXPLORE', member)
        self._visits_reset(member)
    def _member_on_exploit_replaced(self, member):
        self._debug_message('REPLACE', member)
        self._visits_reset(member)

    def _member_on_step(self, member):
        if self._incr_mode == 'stepped':
            self._visits_incr(member)
            self._debug_message('STEPPED', member)
    def _member_on_used_for_exploit(self, member):
        self._debug_message('EXPLOIT', member)
        if self._incr_mode == 'exploited':
            self._visits_incr(member)
            self._debug_message('EXPLOITED', member)

    def suggest(self, filtered: List['IMember']) -> IMember:
        scores = np.array([m.score for m in filtered])
        # normalise scores
        s_min, s_max = np.min(scores), np.max(scores)
        scores = (scores - s_min) / (s_max - s_min + np.finfo(float).eps)
        # step counts
        # HAS ALWAYS TAKEN AT LEAST ONE STEP - EACH MEMBER IS ALREADY VALIDATED
        steps = np.array([self._visits_get(m) for m in filtered]) + 1
        total_steps = np.sum(steps)
        # ucb scores
        # we increment the step count by one because everything has already been visited by default
        ucb_scores = SuggestUcb.ucb1(scores, steps, total_steps, C=self._c)
        ucb_ordering = argsorted_random_ties(ucb_scores)[::-1]
        return filtered[ucb_ordering[0]]

    @staticmethod
    def ucb1(X_i, n_i, n, C=1.):
        return X_i + C * np.sqrt(np.log(n) / n_i)

class SuggestEpsilonGreedy(_SuggestRandomGreedy):
    def __init__(self, epsilon=0.5):
        super().__init__(SuggestUniformRandom(), epsilon)

class SuggestMaxBoltzmann(_SuggestRandomGreedy):
    def __init__(self, epsilon=0.5, temperature=1.0, normalise=True):
        # Epsilon-Softmax is (Random + Softmax)
        # Max Boltzmann Exploration [MBE] is (Greedy + Softmax)
        super().__init__(SuggestSoftmax(temperature=temperature, normalise=normalise), epsilon)

class SuggestEpsilonUcb(_SuggestRandomGreedy):
    def __init__(self, epsilon=0.5, c=1.0, incr_mode='exploited', debug=False):
        # TODO: this name is probably wrong too?
        super().__init__(SuggestUcb(c=c, incr_mode=incr_mode, debug=debug), epsilon)


# ========================================================================= #
# ACCEPT STRATEGIES                                                         #
# ========================================================================= #


class IExploitStrategy(PopulationListener):
    def block(self, current: 'IMember', population: 'IPopulation') -> bool:
        return False
    def accept(self, suggestion: 'IMember', current: 'IMember', population: 'IPopulation'):
        return True
    def filter(self, population: 'IPopulation') -> List['IMember']:
        return list(population)

    def __str__(self):
        return self.__class__.__name__
    def __repr__(self):
        return str(self)

class ExploitStrategyBinaryTournament(IExploitStrategy):
    def accept(self, suggestion: 'IMember', current: 'IMember', population: 'IPopulation'):
        return suggestion.score > current.score

class ExploitStrategyTTestSelection(IExploitStrategy):
    def __init__(self, confidence=0.95):
        assert 0 < confidence < 1  # 1 or 0 are probably errors
        self._confidence = confidence
    def accept(self, suggestion: 'IMember', current: 'IMember', population: 'IPopulation'):
        # TODO, step score history is not the same as exploit score history...
        raise NotImplementedError('Implement Me')

class ExploitStrategyTruncationSelection(IExploitStrategy):
    def __init__(self, bottom_ratio=0.2, top_ratio=0.2, limit_exploits=False):
        assert 0 <= bottom_ratio <= 1
        assert 0 <= top_ratio <= 1
        self._bottom_ratio = bottom_ratio
        self._top_ratio = top_ratio
        # assumes that block is always called before filter.
        self._temp_sorted = None
        self._limit_exploits = limit_exploits
        # RESET EVERY POPULATION STEP
        self._exploited_count = 0

    # POPULATION LISTENERS
    def _population_stepped(self) -> NoReturn:
        self._exploited_count = 0
    def _member_on_used_for_exploit(self, member) -> NoReturn:
        self._exploited_count += 1

    def filter(self, population: 'IPopulation') -> List['IMember']:
        # USE CACHED SORT RESULT
        members, self._temp_sorted = self._temp_sorted, None
        # Top % of the population
        idx_hgh = len(population) - max(1, int(len(population) * self._top_ratio))
        return members[idx_hgh:]

    def block(self, current: 'IMember', population: 'IPopulation'):
        # If the current agent is in the bottom % of the population
        idx_low = max(1, int(len(population) * self._bottom_ratio))

        # dont exploit more than the allowed number each step of the population.
        if self._limit_exploits and self._exploited_count >= idx_low:
            return True

        members = sorted_random_ties(population, key=lambda m: m.score)
        is_blocked = members.index(current) >= idx_low
        # CACHE SORT RESULT IF NEEDED
        self._temp_sorted = None if is_blocked else members
        # return
        return is_blocked

# ========================================================================= #
# GENERALISED EXPLOITER                                                     #
# ========================================================================= #

class GeneralisedExploiter(Exploiter):

    def __init__(self, strategy, suggester=None):
        super().__init__()
        if suggester is None:
            suggester = SuggestUniformRandom()
        assert isinstance(strategy, IExploitStrategy)
        assert isinstance(suggester, ISuggest)
        self._suggester = suggester
        self._strategy = strategy
        # add listeners
        self._listeners = MergedPopulationListener(self._suggester, self._strategy)
        self._listeners.assign_listeners_to(self)

    def exploit(self, population: 'IPopulation', current: 'IMember') -> Optional['IMember']:
        if self._strategy.block(current, population):
            return None
        filtered = self._strategy.filter(population)
        suggestion = self._suggester.suggest(filtered)
        if self._strategy.accept(suggestion, current, population):
            return suggestion
        return None

    def __str__(self):
        return f'{super().__str__()}-{self._strategy}-{self._suggester}'


# ========================================================================= #
# STRATEGIES FROM: Population Based Training of Neural Networks             #
# ========================================================================= #


class OrigExploitTtestSelection(Exploiter):
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


class OrigExploitTruncationSelection(Exploiter):
    """
    From: Population Based Training of Neural Networks
    Section: 4.1.1 PBT for RL
        Truncation selection where we rank all agents in the population by episodic
        reward. If the current agent is in the bottom 20% of the population, we
        sample another agent uniformly from the top 20% of the population, and copy
        its weights and hyper-parameters.
    """

    def __init__(self, bottom_ratio=0.2, top_ratio=0.2):
        super().__init__()
        assert 0 <= bottom_ratio <= 1
        assert 0 <= top_ratio <= 1
        self._bottom_ratio = bottom_ratio
        self._top_ratio = top_ratio

    def exploit(self, population: 'IPopulation', member: 'IMember') -> Optional['IMember']:
        idx_low = max(1, int(len(population) * self._bottom_ratio))
        idx_hgh = len(population) - max(1, int(len(population) * self._top_ratio))

        # we rank all agents in the population by episodic reward (low to high)
        members = sorted_random_ties(population, key=lambda m: m.score)

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


class OrigExploitBinaryTournament(Exploiter):
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
# CUSTOM STRATEGIES - HELPER                                                #
# ========================================================================= #


class _OrigExploitTsSubset(OrigExploitTruncationSelection):
    def __init__(self, bottom_ratio=0.2, top_ratio=0.2, subset_mode='top'):
        super().__init__(bottom_ratio=bottom_ratio, top_ratio=top_ratio)
        assert subset_mode in {'top', 'exclude_bottom', 'all'}
        self._subset_mode = subset_mode

    def _choose_replacement(self, mbrs_low: List['IMember'], mbrs_mid: List['IMember'], mbrs_top: List['IMember'], mbrs: List['IMember'], population: 'IPopulation', member: 'IMember') -> 'IMember':
        if self._subset_mode == 'top':
            subset = mbrs_top
        elif self._subset_mode == 'exclude_bottom':
            subset = mbrs_mid + mbrs_top
        elif self._subset_mode == 'all':
            subset = mbrs[:]
            subset.remove(member)
        else:
            raise KeyError('Invalid subset_mode')
        return self._choose(subset, population, member)

    def _choose(self, subset: List['IMember'], population: 'IPopulation', member: 'IMember'):
        raise NotImplementedError('Override Me')

# ========================================================================= #
# CUSTOM STRATEGIES                                                         #
# ========================================================================= #

class OrigExploitEGreedy(_OrigExploitTsSubset):
    def __init__(self, epsilon=0.5, bottom_ratio=0.2, top_ratio=0.2, subset_mode='top'):
        super().__init__(bottom_ratio=bottom_ratio, top_ratio=top_ratio, subset_mode=subset_mode)
        self._epsilon = epsilon

    def _choose(self, subset: List['IMember'], population: 'IPopulation', member: 'IMember') -> 'IMember':
        if np.random.random() < self._epsilon:
            # with e probability take a random sample (explore) otherwise greedy
            return random.choice(subset)
        else:
            return subset[np.argmax([m.score for m in subset])]

class OrigExploitSoftmax(_OrigExploitTsSubset):
    def __init__(self, temperature=1.0, bottom_ratio=0.2, top_ratio=0.2, subset_mode='top'):
        super().__init__(bottom_ratio=bottom_ratio, top_ratio=top_ratio, subset_mode=subset_mode)
        self._temperature = temperature

    def _choose(self, subset: List['IMember'], population: 'IPopulation', member: 'IMember') -> 'IMember':
        scores = np.array([m.score for m in subset])
        # NORMALISE SCORES BETWEEN 0 AND 1
        s_min, s_max = np.min(scores), np.max(scores)
        scores = (scores - s_min) / (s_max - s_min + np.finfo(float).eps)
        # CALCULATE PROB
        vals = np.exp(scores * self._temperature)
        prob = vals / np.sum(vals)
        # RETURN CATEGORICALLY SAMPLED
        return np.random.choice(subset, p=prob)

class OrigExploitESoftmax(OrigExploitSoftmax):
    def __init__(self, epsilon=0.5, temperature=1.0, bottom_ratio=0.2, top_ratio=0.2, subset_mode='top'):
        super().__init__(temperature=temperature, bottom_ratio=bottom_ratio, top_ratio=top_ratio, subset_mode=subset_mode)
        self._epsilon = epsilon

    def _choose(self, subset: List['IMember'], population: 'IPopulation', member: 'IMember') -> 'IMember':
        if np.random.random() < self._epsilon:
            return super()._choose(subset, population, member)
        else:
            return subset[np.argmax([m.score for m in subset])]


class OrigExploitUcb(_OrigExploitTsSubset):
    def __init__(self, bottom_ratio=0.2, top_ratio=0.2, c=0.1, subset_mode='top', incr_mode='exploited', reset_mode='exploited', select_mode='ucb', normalise_mode='subset', debug=False):
        # >>> high c is BAD
        # >>> low c is BAD
        super().__init__(bottom_ratio=bottom_ratio, top_ratio=top_ratio, subset_mode=subset_mode)
        # UCB
        self.__step_counts = defaultdict(int)
        self._c = c
        # MODES
        assert select_mode in {'ucb', 'ucb_sample', 'uniform'}
        assert incr_mode in {'stepped', 'exploited'}
        assert reset_mode in {'explored_or_exploited', 'explored', 'exploited'}
        assert normalise_mode in {'subset', 'population'}
        self._select_mode = select_mode
        self._incr_mode = incr_mode
        self._reset_mode = reset_mode
        self._normalise_mode = normalise_mode
        # debug
        self.debug = debug

    def _visits_incr(self, member):
        assert isinstance(member, IMember)
        self.__step_counts[member] += 1

    def _visits_reset(self, member):
        assert isinstance(member, IMember)
        self.__step_counts[member] = 0

    def _visits_get(self, member) -> int:
        assert isinstance(member, IMember)
        return self.__step_counts[member]

    def _debug_message(self, message, member=None):
        if self.debug:
            extra = list(self.__step_counts.keys()).index(member) if member and member in self.__step_counts else ''
            tqdm.write(f'{message:>10s}: {list(self.__step_counts.values())} {extra}')

    # THESE TWO STRATEGIES ARE EFFECTIVELY THE SAME
    def _member_on_explored(self, member):
        self._debug_message('EXPLORE', member)
        if self._reset_mode in {'explored_or_exploited', 'explored'}:
            self._visits_reset(member)
            self._debug_message('EXPLORED', member)
    def _member_on_exploit_replaced(self, member):
        self._debug_message('REPLACE', member)
        if self._reset_mode in {'explored_or_exploited', 'exploited'}:
            self._visits_reset(member)
            self._debug_message('REPLACED', member)

    def _member_on_step(self, member):
        if self._incr_mode == 'stepped':
            self._visits_incr(member)
            self._debug_message('STEPPED', member)
    def _member_on_used_for_exploit(self, member):
        self._debug_message('EXPLOIT', member)
        if self._incr_mode == 'exploited':
            self._visits_incr(member)
            self._debug_message('EXPLOITED', member)

    def _choose(self, subset: List['IMember'], population: 'IPopulation', member: 'IMember') -> 'IMember':
        if self.debug:
            tqdm.write('')
            self._debug_message('REPLACING', member)
            tqdm.write('')

        # assert len(_pop) == len(members), 'Not all members were included'
        # normalise scores
        scores = np.array([m.score for m in subset])

        if self._normalise_mode == 'population':
            s_min, s_max = np.min(population.scores), np.max(population.scores)
        elif self._normalise_mode == 'subset':
            s_min, s_max = np.min(scores), np.max(scores)
        else:
            raise KeyError('Invalid normalise_mode')

        scores = (scores - s_min) / (s_max - s_min + np.finfo(float).eps)

        # step counts
        # HAS ALWAYS TAKEN AT LEAST ONE STEP - EACH MEMBER IS ALREADY VALIDATED
        steps = np.array([self._visits_get(m) for m in subset]) + 1
        total_steps = np.sum(steps)

        # ucb scores
        # we increment the step count by one because everything has already been visited by default
        ucb_scores = OrigExploitUcb.ucb1(scores, steps, total_steps, C=self._c)

        # mode
        if self._select_mode == 'ucb':
            ucb_ordering = argsorted_random_ties(ucb_scores)[::-1]
            # ucb_ordering = np.argsort(ucb_scores)[::-1]
            return subset[ucb_ordering[0]]
        elif self._select_mode == 'ucb_sample':
            raise KeyError('ucb_sample is no longer a valid method. TODO: replace with softmax-ucb')
        elif self._select_mode == 'uniform':
            return random.choice(subset)
        else:
            raise KeyError('Invalid select_mode')

    @staticmethod
    def ucb1(X_i, n_i, n, C=1.):
        return X_i + C * np.sqrt(np.log(n) / n_i)


# ========================================================================= #
# END                                                                       #
# ========================================================================= #