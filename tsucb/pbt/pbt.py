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


import random
from typing import List, Iterator, NamedTuple, NoReturn, Optional
import abc
import numpy as np
from tqdm import tqdm
from uuid import uuid4

from tsucb.helper.util import shuffled


# ========================================================================= #
# Population                                                                #
# ========================================================================= #


class IPopulation(abc.ABC):
    """
    Population contains members. A population is evolved over time, using
    Population Based Training, and the methods of exploiting other members
    for better parameters, and then exploring their hyper-parameter space.
    """

    @property
    def best(self) -> 'IMember':
        return max(self.members, key=lambda m: m.eval(self.options))

    def __getitem__(self, item) -> 'IMember':
        return self.members.__getitem__(item)

    def __len__(self) -> int:
        return self.members.__len__()

    def __iter__(self) -> Iterator['IMember']:
        return self.members.__iter__()

    def __contains__(self, member):
        return self.members.__contains__(member)

    @property
    def debug(self):
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def members(self) -> List['IMember']:
        """
        :return: The list of members belonging to the population.
        """
        pass

    @property
    def members_sorted(self):
        """
        :return: The list of members belonging to the population,
                 but sorted in decreasing order of their score.
        """
        return sorted(self.members, key=lambda m: m.score, reverse=True)

    @property
    def scores(self):
        return np.array([m.score for m in self.members])

    @property
    def scores_history(self):
        return np.array([[h.p for h in m] for m in self.members])

    @property
    @abc.abstractmethod
    def options(self) -> dict:
        """
        Used to pass data to IMembers.
        :return: The ditionary of options set for the population.
        """
        pass

    @property
    @abc.abstractmethod
    def exploiter(self) -> 'IExploiter':
        """
        :return: The exploiter.
        """
        pass

    @abc.abstractmethod
    def train(self, n: int, exploit: bool, explore: bool) -> 'IPopulation':
        """
        The Population Based Training algorithm described in the original DeepMind paper.
            [exploit=T, explore=T] = PBT
            [exploit=F, explore=F] = grid-search (optimisation via steps only)
        :param n: Number of training steps to perform for each member.
        :param exploit: Do the exploit phase.
        :param explore: Do the explore phase.
        :return: The member with the best score/score after training.
        """
        pass


class Population(IPopulation):
    """
    Basic implementation of a Population, without providing any abstract methods.
    Use the options to pass data to members.
    """

    def __init__(self, members, exploiter, options=None):
        self._members = members
        if len(members) < 1:
            raise RuntimeError('Population must have at least one member')

        if options is None:
            options = {}
        self._options = options

        assert isinstance(exploiter, IExploiter)
        self._exploiter = exploiter
        self._exploiter._set_used()

        self._debug = options.get('debug', False)

        # unique population identifier
        self._id = uuid4()

        # initialise members
        for i, m in enumerate(self._members):
            m._initialise(self._id, i)


    @property
    def id(self) -> str:
        return self._id

    @property
    def members(self) -> List['IMember']:
        # returns a shallow copy
        return self._members[:]

    @property
    def debug(self):
        return self._debug

    @property
    def options(self) -> dict:
        return self._options

    @property
    def exploiter(self) -> 'IExploiter':
        return self._exploiter

    def train(self, n=None, exploit=True, explore=True, show_progress=True, randomize_order=True) -> 'IPopulation':
        """
        Based on:
        + The original paper
        + https://github.com/angusfung/population-based-training/blob/master/pbt.py
        + https://github.com/BlackHC/pbt/blob/master/notebooks/pbt_first_iteration.ipynb
        :param n: Number of training steps to perform for each member.
        :param exploit: Do the exploit phase.
        :param explore: Do the explore phase.
        :return: The member with the best score/score after training.
        """

        if n is None:
            n = self.options.get('steps', 100)

        itr = tqdm(range(n), 'steps') if show_progress else range(n)

        # TODO: loops should be swapped for async operations
        #       - original paper describes unsyncronised operations, so members can
        #         exploit explore at any time if the conditions are met.
        for i in itr:
            # partial async simulation with members not finishing in the same order.
            ids_members = shuffled(enumerate(self.members), enabled=randomize_order)
            if show_progress:
                max_score = max(m.score for m in self)

            # ADVANCE POPULATION
            for j, (dx, member) in enumerate(ids_members):  # should be async
                if show_progress:
                    itr.set_description(f'step {i+1}/{n} (member {j+1}/{len(self.members)}) [{max_score}]')
                # one step of optimisation using hyper-parameters h
                self._step(member)
                # current model evaluation
                self._eval(member)

            # EXPLOIT / EXPLORE
            for j, (dx, member) in enumerate(ids_members):  # should be async
                if show_progress:
                    itr.set_description(f'step {i + 1}/{n} (exploiting+exploring)')
                # READY?
                if self._is_ready(member):
                    do_explore = explore
                    # EXPLOIT
                    if exploit:
                        if self.debug:
                            tqdm.write(f'[EXPLOITING]: {member.id}')
                        # replace the member using the rest of population to find a better solution
                        exploited = self._exploit(member)
                        # only explore if we exploited
                        do_explore = do_explore and exploited
                    # EXPLORE
                    if do_explore:
                        if self.debug:
                            tqdm.write(f'[EXPLORING]: {member.id}')
                        # produce new hyper-parameters h and update member
                        self._explore(member)
                        # new model evaluation
                        self._eval(member)

            # STEP - END
            self._exploiter._population_stepped()

            # STEP - LOGGING
            if self._options.get('print_scores', False) or self._debug:
                tqdm.write(f'[RESULTS]: step={i+1} max_score={max(m.score for m in self)} min_score={min(m.score for m in self)}')
                for j, m in enumerate(self.members):
                    tqdm.write(f'  {j}: {m.score:5f}{"" if m.history[-1].exploit_id is None else f" <- {m.history[-1].exploit_id}"} | {m.mutable_str}')
                tqdm.write('')

        # TRAINING CLEANUP
        for m in self.members:
            m.cleanup()

        return self

    def _step(self, member) -> NoReturn:
        member.step(self.options)
        self.exploiter._member_on_step(member)

    def _eval(self, member) -> NoReturn:
        member.eval(options=self.options)

    def _is_ready(self, member) -> bool:
        return member.is_ready(self)

    def _exploit(self, member) -> bool:
        exploited_member = member.exploit(self)
        if exploited_member is not None:
            self.exploiter._member_on_used_for_exploit(exploited_member)
            self.exploiter._member_on_exploit_replaced(member)
            if self.debug:
                tqdm.write(f'[EXPLOIT]: {member.id} <- {exploited_member.id}')
            return True
        return False

    def _explore(self, member) -> bool:
        explored = member.explore(self)
        if explored:
            self.exploiter._member_on_explored(member)
        return explored

    def print_history(self):
        n = min(len(m) for m in self.members)
        histories = [[m[i] for m in self.members] for i in range(n)]

        liniage = [i for i in range(len(self))]
        curr_liniage = len(self)

        tqdm.write('')
        for step in histories:
            assert all(h1.t == h2.t for h1, h2 in zip(step[:-1], step[1:]))

            for i, h in enumerate(step):
                if h.exploit_id is not None:
                    liniage[i] = curr_liniage
                    curr_liniage += 1

            string_a = ''
            string_b = ''
            for l, h in zip(liniage, step):
                string_a += f'{str(l):>3s}: {str(np.around(h.p, 2)):<5s} '
                string_b += f'    {f"({h.exploit_id})":6s} ' if (h.exploit_id is not None) else ' '*11

            tqdm.write(f'{str(step[0].t):4s}| {string_a}')
            tqdm.write(f'    | {string_b}')
        tqdm.write('')


# ========================================================================= #
# History                                                                   #
# ========================================================================= #


class HistoryItem(NamedTuple):
    """
    Stores the state of a member at a particular time step.
    """

    """ hyper-parameters """
    h: object
    """ score """
    p: float
    """ step number """
    t: int
    """ is exploited """
    exploit_id: Optional[int]
    """ result """
    result: Optional[object]


# ========================================================================= #
# Member                                                                    #
# ========================================================================= #


class IMember(abc.ABC):
    """
    Members belong to a population and provide mechanisms to
    step, evaluate, explore, and exploit the underlying
    machine learning algorithm.

    Members remember their history of updates.
    """

    def __init__(self, *args, **kwargs):
        self._id = None
        self._population_id = None
        # params
        self._args = args
        self._kwargs = kwargs

    @abc.abstractmethod
    def copy_h(self) -> object:
        """
        :return: A deepcopy of the members hyper-parameters (h)
        """
        pass

    @property
    def mutable_str(self) -> str:
        raise NotImplementedError('Override Me')

    @abc.abstractmethod
    def _set_h(self, h) -> NoReturn:
        pass

    @abc.abstractmethod
    def _save_theta(self, id):
        pass

    @abc.abstractmethod
    def _load_theta(self, id):
        pass

    @property
    def id(self):
        assert self._id is not None, 'Member not yet added to a population'
        return self._id

    @property
    def population_id(self):
        assert self._population_id is not None, 'Member not yet added to a population'
        return self._population_id

    def _initialise(self, population_id, id):
        assert self._id is None, 'Member already added to a population'
        assert population_id is not None, 'Invalid Population id'
        assert id is not None, 'Invalid id'
        self._population_id = population_id
        self._id = id
        self._setup(*self._args, **self._kwargs)
        del self._args
        del self._kwargs

    @abc.abstractmethod
    def _setup(self, *args, **kwargs) -> NoReturn:
        """
        Intended to be overridden to initialise the member.
        """
        pass

    @property
    @abc.abstractmethod
    def history(self) -> List['HistoryItem']:
        pass

    # def __getitem__(self, item) -> 'HistoryItem':
    #     return self.history.__getitem__(item)
    #
    # def __len__(self) -> int:
    #     return self.history.__len__()
    #
    # def __iter__(self) -> Iterator['HistoryItem']:
    #     return self.history.__iter__()

    @abc.abstractmethod
    def step(self, options: dict) -> NoReturn:
        """
        Perform one step or batch of the optimisation process on the underlying
        machine learning algorithm. The values of this members parameters (theta)
        should be updated, for example via gradient decent.
        :param options: A dictionary of options set for the population, values not guaranteed to exist.
        """
        pass

    @abc.abstractmethod
    def eval(self, options: dict) -> float:
        """
        Evaluate the score of the underlying machine learning algorithm.
        :param options: A dictionary of options set for the population, values not guaranteed to exist.
        :return: A float value indicating the current score.
        """
        pass

    @property
    @abc.abstractmethod
    def score(self):
        """
        :return: The score from the last evaluation
        """
        pass

    @abc.abstractmethod
    def is_ready(self, population: 'IPopulation') -> bool:
        """
        :param population: The population that this member belongs to.
        :return: True if this member is ready for the exploit/explore stage.
        """
        pass

    @abc.abstractmethod
    def exploit(self, population: 'IPopulation') -> Optional['IMember']:
        """
        Update the parameters (theta) of this member with those of
        another member, using various selection strategies.
        :param population: The population that this member belongs to.
        :return The member chosen to replace this one. None if no replacement happened.
        """
        pass

    @abc.abstractmethod
    def explore(self, population: 'IPopulation') -> bool:
        """
        Update the hyper-parameters (h) of this member using various
        approaches or perturbation strategies.
        :param population: The population that this member belongs to.
        :return: True if this members hyper-parameters were successfully explored and updated.
        """
        pass

    @abc.abstractmethod
    def cleanup(self):
        """
        Called after training, intended to destroy resources held by the member.
        """


class Member(IMember):
    """
    Basic implementation of member, implementing the common functionality
    required by the PBT algorithm, and most subclasses. Can handle any data
    type for the parameters and hyper-parameters. Only evaluates the score
    once per step using caching.

    Provides simplified abstract methods that need to be overridden.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # current score, also known as 'Q'
        self._p = float('-inf')
        # current step
        self._t = 0
        self._t_last_explore = 0
        # should the score value be recalculated
        self._recal = False  # we want to use infinity for the first step.
        self._exploited = None
        self._results = None
        # other vars
        self._history = []
        self._id = None

    def __str__(self):
        """ Get the string representation of the member """
        return f"{self._t} : {self._p}"

    @property
    def history(self) -> List['HistoryItem']:
        return self._history

    def _push_history(self):
        self._history.append(HistoryItem(self.copy_h(), self._p, self._t, self._exploited, self._results))
        self._exploited = None

    def step(self, options: dict):
        # append to member's history.
        self._push_history()
        # update the parameters of the member.
        self._results = self._step(options)
        self._save_theta(self.id)
        # indicate that the score score should be recalculated.
        self._recal = True
        # increment step counter
        self._t += 1

    def eval(self, options: dict):
        # only recalculate score score if necessary
        if self._recal:
            old_score = self._p
            self._p = self._eval(options)
            self._recal = False
            if options.get('debug', False):
                tqdm.write(f'\n[RECALC]: {self.id}: {old_score} -> {self._p}')
        return self._p

    @property
    def score(self):
        if self._t <= 0:
            return float('-inf')
        assert not self._recal, f'The member {self.id} has not been evaluated'
        return self._p

    def is_ready(self, population: 'IPopulation') -> bool:
        ready = self._is_ready(population, self._t, self._t - self._t_last_explore)
        if population.debug:
            tqdm.write(f'[READY]: {self.id} {ready} {self.score}')
        return ready

    def exploit(self, population: 'IPopulation') -> Optional['IMember']:
        member = population.exploiter.exploit(population, self)
        # Skip exploit if None
        if member is None:
            return None
        # Skip exploit is the Same
        if member is self:
            if population._options.get('warn_exploit_self', False):
                tqdm.write(f"[WARNING]: {self.id} Exploited itself - [{population.exploiter}]")
            return None
        # Copy parameters & hyperparameters
        self._load_theta(member.id)
        self._save_theta(self.id)
        self._set_h(member.copy_h())
        self._recal = True
        # Append to history on next step
        self._exploited = member.id
        # Success
        return member

    def explore(self, population: 'IPopulation') -> bool:
        exploring_h = self._explored_h(population)
        assert exploring_h is not None, "Explore values are None, problem with explorer?"
        # Set hyperparameters
        self._set_h(exploring_h)
        self._recal = True
        self._t_last_explore = self._t
        # Success
        return True

    def _is_ready(self, population: 'IPopulation', steps: int, steps_since_explore: int) -> bool:
        """
        :param population: The population that this member belongs to.
        :param steps: The number of steps performed.
        :param steps_since_explore: The number of steps since the last hyper-parameter change.
        :return: True if this member is ready for the exploit/explore stage.
        """
        return steps_since_explore >= population.options['steps_till_ready']

    @abc.abstractmethod
    def _step(self, options: dict) -> object:
        """
        :param options: A dictionary of options set for the population, values not guaranteed to exist.
        :return: Results generated by the step, to be saved in the history
        """
        pass

    @abc.abstractmethod
    def _eval(self, options: dict) -> float:
        """
        :param options: A dictionary of options set for the population, values not guaranteed to exist.
        :return: A float value indicating the current score, used by self.eval().
        """
        pass

    @abc.abstractmethod
    def _explored_h(self, population: 'IPopulation') -> object:
        """
        :param population: The population that this member belongs to.
        :return: Updated hyper-parameters, used by self.explore().
                 *NB* this should be a new instance.
        """
        pass

    def cleanup(self):
        self._cleanup()

    def _cleanup(self):
        """ OVERRIDE ME """
        pass


# ========================================================================= #
# Exploiter                                                                  #
# ========================================================================= #


class PopulationListener(abc.ABC):
    def _member_on_step(self, member) -> NoReturn:
        pass
    def _member_on_explored(self, member) -> NoReturn:
        pass
    def _member_on_exploit_replaced(self, member) -> NoReturn:
        pass
    def _member_on_used_for_exploit(self, member) -> NoReturn:
        pass
    def _population_stepped(self) -> NoReturn:
        pass

    # HELPER
    def assign_listeners_to(self, target):
        target._member_on_step = self._member_on_step
        target._member_on_explored = self._member_on_explored
        target._member_on_exploit_replaced = self._member_on_exploit_replaced
        target._member_on_used_for_exploit = self._member_on_used_for_exploit
        target._population_stepped = self._population_stepped
        return self


class MergedPopulationListener(PopulationListener):
    def __init__(self, a, b):
        self._a = a
        self._b = b
    def _member_on_step(self, member) -> NoReturn:
        self._a._member_on_step(member)
        self._b._member_on_step(member)
    def _member_on_explored(self, member) -> NoReturn:
        self._a._member_on_explored(member)
        self._b._member_on_explored(member)
    def _member_on_exploit_replaced(self, member) -> NoReturn:
        self._a._member_on_exploit_replaced(member)
        self._b._member_on_exploit_replaced(member)
    def _member_on_used_for_exploit(self, member) -> NoReturn:
        self._a._member_on_used_for_exploit(member)
        self._b._member_on_used_for_exploit(member)
    def _population_stepped(self) -> NoReturn:
        self._a._population_stepped()
        self._b._population_stepped()


class IExploiter(PopulationListener):

    def __init__(self):
        self._used = False

    def _set_used(self):
        assert not self._used, 'This exploiter as already been used.'
        self._used = True

    @abc.abstractmethod
    def exploit(self, population: 'IPopulation', member: 'IMember') -> Optional['IMember']:
        """
        :param population: The population that this member belongs to.
        :param member: The current member requiring exploitation.
        :return The member to be exploited, used by self.exploit().
        """
        pass

    @property
    def name(self):
        n = self.__class__.__name__
        if len(n) > 10 and n.lower().startswith('exploiter'):
            n = n[9:]
        elif len(n) > 8 and n.lower().startswith('exploit'):
            n = n[7:]
        return n

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return str(self)


class Exploiter(IExploiter):
    pass
