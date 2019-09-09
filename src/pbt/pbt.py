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


from typing import List, Iterator, NamedTuple, NoReturn, Optional
import abc
import numpy as np


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

    @property
    def members(self) -> List['IMember']:
        # returns a shallow copy
        return self._members[:]

    @property
    def options(self) -> dict:
        return self._options

    @property
    def exploiter(self) -> 'IExploiter':
        return self._exploiter

    def train(self, n=None, exploit=True, explore=True) -> 'IPopulation':
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

        # TODO: loops should be swapped for async operations
        for i in range(n):
            for idx, member in enumerate(self.members):  # should be async

                # one step of optimisation using hyper-parameters h
                self._step(member)
                # current model evaluation
                self._eval(member)

                if self._is_ready(member):
                    do_explore = explore

                    if exploit:
                        # replace the member using the rest of population to find a better solution
                        exploited = self._exploit(member)
                        # only explore if we exploited
                        do_explore = do_explore and exploited

                    if do_explore:
                        # produce new hyper-parameters h and update member
                        self._explore(member)
                        # new model evaluation
                        self._eval(member)

                # update population
                # TODO: needed for async operations
                # self._update(idx, member)

        return self

    def _step(self, member) -> NoReturn:
        member.step(self.options)
        self.exploiter._member_on_step(member)

    def _eval(self, member) -> NoReturn:
        member.eval(options=self.options)

    def _is_ready(self, member) -> bool:
        return member.is_ready(self)

    def _exploit(self, member) -> bool:
        exploited = member.exploit(self)
        self.exploiter._member_on_exploited(member)
        return exploited

    def _explore(self, member) -> bool:
        explored = member.explore(self)
        self.exploiter._member_on_explored(member)
        return explored




# ========================================================================= #
# History                                                                   #
# ========================================================================= #


class HistoryItem(NamedTuple):
    """
    Stores the state of a member at a particular time step.
    """

    """ hyper-parameters """
    h: object
    """ parameters """
    theta: object
    """ score """
    p: float
    """ step number """
    t: int
    """ is exploited """
    exploit_id: Optional[int]


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

    @abc.abstractmethod
    def copy_h(self) -> object:
        """
        :return: A deepcopy of the members hyper-parameters (h)
        """
        pass

    @abc.abstractmethod
    def copy_theta(self) -> object:
        """
        :return: A deepcopy of the members parameters (theta)
        """
        pass
    
    @property
    @abc.abstractmethod
    def history(self) -> List['HistoryItem']:
        pass

    def __getitem__(self, item) -> 'HistoryItem':
        return self.history.__getitem__(item)

    def __len__(self) -> int:
        return self.history.__len__()

    def __iter__(self) -> Iterator['HistoryItem']:
        return self.history.__iter__()

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
    def exploit(self, population: 'IPopulation') -> bool:
        """
        Update the parameters (theta) of this member with those of
        another member, using various selection strategies.
        :param population: The population that this member belongs to.
        :return True if another member was successfully exploited.
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


class Member(IMember):
    """
    Basic implementation of member, implementing the common functionality
    required by the PBT algorithm, and most subclasses. Can handle any data
    type for the parameters and hyper-parameters. Only evaluates the score
    once per step using caching.

    Provides simplified abstract methods that need to be overridden.
    """

    def __init__(self, h=None, theta=None, p=float('-inf'), t=0):
        # model hyper-parameters
        self._h = h
        # model parameters
        self._theta = theta
        # check hyper-parameters and parameters are valid.
        if (self._h is None) or (self._theta is None):
            raise RuntimeError('hyper-parameters and parameters must be set')
        # current score, also known as 'Q'
        self._p = p
        # current step
        self._t = t
        # should the score value be recalculated
        self._recal = False
        self._exploited = None
        # other vars
        self._history = []

    def __str__(self):
        """ Get the string representation of the member """
        return f"{self._t} : {self._p} : {self._h} : {self._theta}"

    @property
    def history(self) -> List['HistoryItem']:
        return self._history

    def step(self, options: dict):
        # append to member's history.
        self._history.append(HistoryItem(self.copy_h(), self.copy_theta(), self._p, self._t, self._exploited))
        self._exploited = None
        # update the parameters of the member.
        self._theta = self._step(options)
        # indicate that the score score should be recalculated.
        self._recal = True
        # increment step counter
        self._t += 1
        return self._t

    def eval(self, options: dict):
        # only recalculate score score if necessary
        if self._recal:
            self._p = self._eval(options)
            self._recal = False
        return self._p

    @property
    def score(self):
        return self._p

    def is_ready(self, population: 'IPopulation') -> bool:
        return self._is_ready(population)

    def exploit(self, population: 'IPopulation') -> bool:
        member = population.exploiter.exploit(population, self)
        # only use the exploited member's parameters if it is not the same.
        if member is not None:
            if self != member:
                self._theta = member.copy_theta()
                self._exploited = population.members.index(member)
                return True
            else:
                print(f"Member exploited itself, problem with exploiter? {type(population.exploiter).__name__}")
                return False
        return False

    def explore(self, population: 'IPopulation') -> bool:
        exploring_h = self._explore(population)
        # only use the explored hyper-parameters if they are valid.
        if exploring_h is not None:
            self._h = exploring_h
            return True
        else:
            print("Explore values are None, problem with exploration exploiter?")
            return False

    @abc.abstractmethod
    def _is_ready(self, population: 'IPopulation') -> bool:
        """
        :param population: The population that this member belongs to.
        :return: True if this member is ready for the exploit/explore stage.
        """
        pass

    @abc.abstractmethod
    def _step(self, options: dict) -> object:
        """
        :param options: A dictionary of options set for the population, values not guaranteed to exist.
        :return: Updated values of the parameters, used by self.step()
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
    def _explore(self, population: 'IPopulation') -> bool:
        """
        :param population: The population that this member belongs to.
        :return: Updated hyper-parameters, used by self.explore().
        """
        pass


# ========================================================================= #
# Exploiter                                                                  #
# ========================================================================= #


class IExploiter(abc.ABC):

    @abc.abstractmethod
    def exploit(self, population: 'IPopulation', member: 'IMember') -> Optional['IMember']:
        """
        :param population: The population that this member belongs to.
        :param member: The current member requiring exploitation.
        :return The member to be exploited, used by self.exploit().
        """
        pass

    def _member_on_step(self, member) -> NoReturn:
        pass

    def _member_on_explored(self, member) -> NoReturn:
        pass

    def _member_on_exploited(self, member) -> NoReturn:
        pass


class Exploiter(IExploiter):
    pass
