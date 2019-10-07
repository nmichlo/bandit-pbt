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
import argparse
from uuid import uuid4

from tsucb.helper import util
from tsucb.helper.attrs import field, Attrs
from tsucb.pbt.pbt import IExploiter, Population
from tsucb.pbt.strategies import *

# ========================================================================= #
# HELPER                                                                    #
# ========================================================================= #

# VALIDATORS
def val_range(a, b, number_type=float):
    assert number_type in {int, float}
    def inner(x):
        x = number_type(x)
        if not (a <= x <= b):
            raise argparse.ArgumentTypeError(f'{x} not in range [{a}, {b}]')
        return x
    return inner

# HELPER
def make_usage_tracker(obj):
    used = set()
    def use(field):
        used.add(field)
        return getattr(obj, field)
    return use, used

# ========================================================================= #
# UCB EXPERIMENT                                                            #
# ========================================================================= #

INF = float('inf')

DATASET_CHOICES = [
    'MNIST',
    'FashionMNIST',
]

SUGGEST_CHOICES = [
    'random', 'e-greedy',
    'softmax', 'e-softmax',
    'ucb', 'e-ucb'
]

STRATEGY_CHOICES = [
    'ts', 'truncation-selection',
    'bt', 'binary-tournament-selection',
    'tt', 't-test-selection'
]

INCR_MODES = [
    'exploited',
    'stepped',
]

EXPERIMENT_CHOICES = [
    'toy',
    'cnn'
]

class ExperimentArgs(Attrs):

    # EXPERIMENT
    experiment_repeats:       int             = field(default=1,           cast=val_range(1, INF))                     # used
    experiment_name:          str             = field(default=uuid4())                                                 # TODO
    experiment_type:          str             = field(default='toy',       cast=str.lower, choices=EXPERIMENT_CHOICES) # used
    experiment_seed:          Optional[int]   = field(default=None)                                                    # used
    # CNN
    cnn_dataset:              str             = field(default='MNIST',     choices=DATASET_CHOICES)                    # used
    cnn_batch_size:           int             = field(default=32,          cast=val_range(1, 1024))                    # used
    cnn_use_cpu:              bool            = field(default=False)                                                    # used
    cnn_step_divs:            int             = field(default=1,           cast=val_range(1, 1000))                    # used
    # PBT
    pbt_print:                bool            = field(default=False)                                                    # used
    pbt_target_steps:         int             = field(default=10,          cast=val_range(1, INF))                     # used
    pbt_target_score:         Optional[float] = field(default=None)                                                    # used
    pbt_members:              int             = field(default=25,          cast=val_range(1, INF))                     # used
    pbt_members_ready_after:  int             = field(default=2,           cast=val_range(1, INF))                     # used
    pbt_exploit_strategy:     str             = field(default='ts',        cast=str.lower, choices=STRATEGY_CHOICES)   # used
    pbt_exploit_suggest:      str             = field(default='random',    cast=str.lower, choices=SUGGEST_CHOICES)    # used
    pbt_disable_exploit:      bool            = field(default=False)                                                    # used
    pbt_disable_explore:      bool            = field(default=False)                                                    # used
    # EXPLOITER - UCB
    suggest_ucb_incr_mode:    str             = field(default='exploited', cast=str.lower, choices=INCR_MODES)         # used
    suggest_ucb_c:            float           = field(default=1.00,        cast=val_range(0.0, 2.0))                   # used
    suggest_softmax_temp:     float           = field(default=1.00,        cast=val_range(0.0, INF))                   # used
    suggest_eps:              float           = field(default=0.75,        cast=val_range(0.0, 1.0))                   # used
    # EXPLOITER - UCB & TS
    strategy_ts_ratio_top:    float           = field(default=0.20,        cast=val_range(0.0, 1.0))                   # used
    strategy_ts_ratio_bottom: float           = field(default=0.20,        cast=val_range(0.0, 1.0))                   # used
    strategy_tt_confidence:   float           = field(default=0.95,        cast=val_range(0.0, 1.0))                   # used
    # EXTRA
    debug:                    bool            = field(default=False)                                                   # used
    enable_comet:             bool            = field(default=False)                                                   # TODO

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._use_suggest_field, self._used_suggest_fields = make_usage_tracker(self)
        self._use_strategy_field, self._used_strategy_fields = make_usage_tracker(self)
        self._use_cnn_field, self._used_cnn_fields = make_usage_tracker(self)

    # -------- #
    # INSTANCE #
    # -------- #

    def to_dict(self, useful_only=False):
        # temp
        self.make_suggest()
        self.make_strategy()
        self.make_member()
        # get dict
        opts = self.as_dict()
        if useful_only:
            opts = {k: v for k, v in opts.items() if k not in {'debug', 'enable_comet'}}
            opts = {k: v for k, v in opts.items() if not k.startswith('suggest_') or k in self._used_suggest_fields}
            opts = {k: v for k, v in opts.items() if not k.startswith('strategy_') or k in self._used_strategy_fields}
            opts = {k: v for k, v in opts.items() if not k.startswith('cnn_') or k in self._used_cnn_fields}
        return opts

    def make_suggest(self) -> 'ISuggest':
        u = self._use_suggest_field
        if self.pbt_exploit_suggest == 'random':
            suggester = SuggestUniformRandom()
        elif self.pbt_exploit_suggest == 'e-greedy':
            suggester = SuggestEpsilonGreedy(epsilon=u('suggest_eps'))
        elif self.pbt_exploit_suggest == 'softmax':
            suggester = SuggestSoftmax(temperature=u('suggest_softmax_temp'))
        elif self.pbt_exploit_suggest == 'e-softmax':
            suggester = SuggestEpsilonSoftmax(epsilon=u('suggest_eps'), temperature=u('suggest_softmax_temp'))
        elif self.pbt_exploit_suggest == 'ucb':
            suggester = SuggestUcb(c=u('suggest_ucb_c'), incr_mode=u('suggest_ucb_incr_mode'))
        elif self.pbt_exploit_suggest == 'e-ucb':
            suggester = SuggestEpsilonUcb(epsilon=u('suggest_eps'), c=u('suggest_ucb_c'), incr_mode=u('suggest_ucb_incr_mode'))
        else:
            raise KeyError(f'Invalid suggester: {self.pbt_exploit_suggest}')
        return suggester

    def make_strategy(self) -> 'IExploitStrategy':
        u = self._use_strategy_field
        if self.pbt_exploit_strategy in {'ts', 'truncation-selection'}:
            strategy = ExploitStrategyTruncationSelection(bottom_ratio=u('strategy_ts_ratio_bottom'), top_ratio=u('strategy_ts_ratio_top'))
        elif self.pbt_exploit_strategy in {'bt', 'binary-tournament-selection'}:
            strategy = ExploitStrategyBinaryTournament()
        elif self.pbt_exploit_strategy in {'tt', 't-test-selection'}:
            strategy = ExploitStrategyTTestSelection(confidence=u('strategy_tt_confidence'))
        else:
            raise KeyError(f'Invalid suggester: {self.pbt_exploit_suggest}')
        return strategy

    def make_exploiter(self) -> 'IExploiter':
        return GeneralisedExploiter(
            strategy=self.make_strategy(),
            suggester=self.make_suggest(),
        )

    def make_member(self):
        if self.experiment_type == 'toy':
            from tsucb.pbt.examples.pbt_paper_toy_example import ToyMember, ToyHyperParams
            return ToyMember(h=ToyHyperParams(coef=np.random.rand(2) * 0.5, alpha=0.01), theta=np.array([.9, .9]))
        elif self.experiment_type == 'cnn':
            from tsucb.pbt.examples.pbt_local_mnist_example import MemberTorch, random_uniform, random_log_uniform
            u = self._use_cnn_field
            return MemberTorch(config=dict(
                model='example', loss='NLLLoss', optimizer='SGD',
                dataset=u('cnn_dataset'),
                model_options={}, dataset_options={}, loss_options={}, optimizer_options=dict(
                    lr=random_log_uniform(0.0001, 0.1),
                    momentum=random_uniform(0.01, 0.99),
                ),
                mutations={
                    'optimizer_options/lr':       ('uniform_perturb', 0.5,  1.8, 0.0001, 0.10),  # eg. 0.8 < 1/1.2  shifts exploration towards getting smaller
                    'optimizer_options/momentum': ('uniform_perturb', 0.5, 2.00, 0.0100, 0.99),  #     0.8 = 1/1.25 is balanced
                },
                train_images_per_step=60000//u('cnn_step_divs'),
                batch_size=u('cnn_batch_size'),
                use_gpu=not u('cnn_use_cpu'),
            ))
        else:
            raise KeyError(f'Invalid experiment_type: {self.experiment_type}')

    def make_members(self) -> List['IMember']:
        return [self.make_member() for _ in range(self.pbt_members)]

    def make_population(self) -> 'IPopulation':
        return Population(
            members=self.make_members(),
            exploiter=self.make_exploiter(),
            options=dict(
                steps_till_ready=self.pbt_members_ready_after,
                steps=self.pbt_target_steps,
                target_score=self.pbt_target_score,
                debug=self.debug,
                warn_exploit_self=True,
                print_scores=self.pbt_print,
            )
        )

    def do_training_run(self, seed=None):
        util.seed(self.experiment_seed if (seed is None) else seed)
        population = self.make_population()
        population.train(
            n=self.pbt_target_steps,
            exploit=not self.pbt_disable_exploit,
            explore=not self.pbt_disable_explore,
        )
        return population

    def do_experiment(self, seed=None):
        seed = self.experiment_seed if (seed is None) else seed
        for i in tqdm(range(self.experiment_repeats), 'repeat'):
            population = self.do_training_run(None if (seed is None) else seed+i)


print(ExperimentArgs().to_dict())
print(ExperimentArgs().to_dict(useful_only=True))


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
