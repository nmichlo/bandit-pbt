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


from typing import List, Iterator, NamedTuple, NoReturn
import abc


class IPopulation(abc.ABC):
    """
    Population contains members. A population is evolved over time, using
    Population Based Training, and the methods of exploiting other members
    for better parameters, and then exploring their hyper-parameter space.
    """

    @property
    def best(self):
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
    @abc.abstractmethod
    def options(self) -> dict:
        """
        Used to pass data to IMembers.
        :return: The ditionary of options set for the population.
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
        :return: The member with the best score/performance after training.
        """
        pass


class Population(IPopulation):
    """
    Basic implementation of a Population, without providing any abstract methods.
    Use the options to pass data to members.
    """

    def __init__(self, members, options=None):
        self._members = members
        if len(members) < 1:
            raise RuntimeError('Population must have at least one member')

        self._options = options
        if self._options is None:
            self._options = {}

    @property
    def members(self) -> List['IMember']:
        return self._members

    @property
    def options(self) -> dict:
        return self._options

    def train(self, n=None, exploit=True, explore=True) -> 'Population':
        """
        Based on:
        + The original paper
        + https://github.com/angusfung/population-based-training/blob/master/pbt.py
        + https://github.com/BlackHC/pbt/blob/master/notebooks/pbt_first_iteration.ipynb
        :param n: Number of training steps to perform for each member.
        :param exploit: Do the exploit phase.
        :param explore: Do the explore phase.
        :return: The member with the best score/performance after training.
        """

        if n is None:
            n = self.options.get('steps', 100)

        # TODO: loops should be swapped for async operations
        for i in range(n):
            for idx, member in enumerate(self.members):  # should be async

                # one step of optimisation using hyper-parameters h
                member.step(options=self.options)
                # current model evaluation
                member.eval(options=self.options)

                if member.is_ready(self):
                    if exploit:
                        # replace the member using the rest of population to find a better solution
                        member.exploit(self)
                    if explore:
                        # produce new hyper-parameters h and update member
                        member.explore(self)
                        # new model evaluation
                        member.eval(options=self.options)

                # update population
                # TODO: needed for async operations
                # self._update(idx, member)

        return self


class HistoryItem(NamedTuple):
    """
    Stores the state of a member at a particular time step.
    """

    """ hyper-parameters """
    h: object
    """ parameters """
    theta: object
    """ performance """
    p: float
    """ step number """
    t: int


class IMember(abc.ABC):
    """
    Members belong to a population and provide mechanisms to
    step, evaluate, explore, and exploit the underlying
    machine learning algorithm.

    Members remember their history of updates.
    """

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
        Evaluate the performance of the underlying machine learning algorithm.
        :param options: A dictionary of options set for the population, values not guaranteed to exist.
        :return: A float value indicating the current performance.
        """
        pass

    @abc.abstractmethod
    def is_ready(self, population: 'Population') -> bool:
        """
        :param population: The population that this member belongs to.
        :return: True if this member is ready for the exploit/explore stage.
        """
        pass

    @abc.abstractmethod
    def exploit(self, population: 'Population') -> bool:
        """
        Update the parameters (theta) of this member with those of
        another member, using various selection strategies.
        :param population: The population that this member belongs to.
        :return True if another member was successfully exploited.
        """
        pass

    @abc.abstractmethod
    def explore(self, population: 'Population') -> bool:
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
    type for the parameters and hyper-parameters. Only evaluates the performance
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
        # current performance, also known as 'Q'
        self._p = p
        # current step
        self._t = t
        # should the performance value be recalculated
        self._recal = False
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
        self._history.append(HistoryItem(self.copy_h(), self.copy_theta(), self._p, self._t))
        # update the parameters of the member.
        self._theta = self._step(options)
        # indicate that the performance score should be recalculated.
        self._recal = True
        # increment step counter
        self._t += 1
        return self._t

    def eval(self, options: dict):
        # only recalculate performance score if necessary
        if self._recal:
            self._p = self._eval(options)
            self._recal = False
        return self._p

    def is_ready(self, population: 'Population') -> bool:
        return self._is_ready(population)

    def exploit(self, population: 'Population') -> bool:
        member = self._exploit(population)
        # only use the exploited member's parameters if it is not the same.
        if member is not None:
            if self != member:
                self._theta = member.copy_theta()
                return True
            else:
                print("Member exploited itself, problem with exploitation strategy?")
                return False
        return False

    def explore(self, population: 'Population') -> bool:
        exploring_h = self._explore(population)
        # only use the explored hyper-parameters if they are valid.
        if exploring_h is not None:
            self._h = exploring_h
            return True
        else:
            print("Explore values are None, problem with exploration strategy?")
            return False

    @abc.abstractmethod
    def copy_h(self) -> object:
        """
        :return: A deepcopy of the members hyper-parameters
        """
        pass

    @abc.abstractmethod
    def copy_theta(self) -> object:
        """
        :return: A deepcopy of the members parameters
        """
        pass

    @abc.abstractmethod
    def _is_ready(self, population: 'Population') -> bool:
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
        :return: A float value indicating the current performance, used by self.eval().
        """
        pass

    @abc.abstractmethod
    def _exploit(self, population: 'Population') -> 'Member':
        """
        :param population: The population that this member belongs to.
        :return The member to be exploited, used by self.exploit().
        """
        pass

    @abc.abstractmethod
    def _explore(self, population: 'Population') -> bool:
        """
        :param population: The population that this member belongs to.
        :return: Updated hyper-parameters, used by self.explore().
        """
        pass