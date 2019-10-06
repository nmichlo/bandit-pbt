
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
from typing import NamedTuple
from uuid import uuid4
from tsucb.pbt.pbt import IExploiter


# ========================================================================= #
# args                                                                      #
# ========================================================================= #


def argparse_number_range(a, b, number_type):
    assert number_type in {int, float}
    def inner(x):
        x = number_type(x)
        if not (a <= x <= b):
            raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0]"%(x,))
        return x
    return inner


# ========================================================================= #
# UCB EXPERIMENT                                                            #
# ========================================================================= #


class UcbExperimentArgs(NamedTuple):
    # EXPERIMENT
    experiment_repeats: int
    experiment_name: str
    # PBT
    pbt_steps: int
    pbt_members: int
    pbt_members_ready_after: int
    pbt_exploiter: str
    pbt_do_exploit: bool
    pbt_do_explore: bool
    # EXPLOITER - UCB
    ucb_select_mode: str
    ucb_incr_mode: str
    ucb_reset_mode: str
    ucb_subset_mode: str
    ucb_normalise_mode: str
    ucb_c: float
    # EXPLOITER - UCB & TS
    ts_ratio_top: float
    ts_ratio_bottom: float
    # EXTRA
    debug: bool
    disable_comet: bool

    @property
    def is_ucb(self):
        return self.pbt_exploiter == 'ucb'

    def to_dict(self, useful_only=False):
        opts = {k: getattr(self, k) for k in self._fields}
        if useful_only:
            if self.pbt_exploiter == 'ucb':
                pass
            elif self.pbt_exploiter == 'ts':
                opts = {k: v for k, v in opts.items() if not k.starts_with('ucb_')}
        return opts

    def make_exploiter(self) -> 'IExploiter':
        if self.pbt_exploiter == 'ucb':
            from tsucb.pbt.strategies import ExploitUcb
            return ExploitUcb(
                bottom_ratio=self.ts_ratio_bottom,
                top_ratio=self.ts_ratio_top,
                c=self.ucb_c,
                subset_mode=self.ucb_subset_mode,
                incr_mode=self.ucb_incr_mode,
                reset_mode=self.ucb_reset_mode,
                select_mode=self.ucb_select_mode,
                normalise_mode=self.ucb_normalise_mode,
                debug=self.debug
            )
        elif self.pbt_exploiter == 'ts':
            from tsucb.pbt.strategies import OrigExploitTruncationSelection
            return OrigExploitTruncationSelection(
                bottom_ratio=self.ts_ratio_bottom,
                top_ratio=self.ts_ratio_top,
            )
        else:
            raise KeyError(f'Invalid exploiter: {self.pbt_exploiter}')


    @staticmethod
    def _create_parser(defaults=None) -> 'argparse.ArgumentParser':
        if defaults is None:
            defaults = {}

        parser = argparse.ArgumentParser()
        # EXPERIMENT
        parser.add_argument('-n', '--experiment-repeats',type=argparse_number_range(1, float('inf'), int),                           default=defaults.get('experiment_repeats', 1))
        parser.add_argument('--experiment-name',         type=str, default=uuid4())
        # PBT
        parser.add_argument('--pbt-steps',               type=argparse_number_range(1, float('inf'), int),                           default=defaults.get('pbt_steps', 50))
        parser.add_argument('--pbt-members',             type=argparse_number_range(1, float('inf'), int),                           default=defaults.get('pbt_members', 10))
        parser.add_argument('--pbt-members-ready-after', type=argparse_number_range(1, float('inf'), int),                           default=defaults.get('pbt_members_ready_after', 2))
        parser.add_argument('--pbt-exploiter',           type=str.lower, choices=['ucb', 'ts'],                                      default=defaults.get('pbt_exploiter', 'ucb'))
        parser.add_argument('--pbt-disable-exploit',     action='store_false', dest='pbt_do_exploit')
        parser.add_argument('--pbt-disable-explore',     action='store_false', dest='pbt_do_explore')
        # EXPLOITER - UCB
        parser.add_argument('--ucb-select-mode',         type=str.lower, choices=['ucb', 'ucb_sample', 'uniform'],                   default=defaults.get('ucb_select_mode', 'ucb'))
        parser.add_argument('--ucb-incr-mode',           type=str.lower, choices=['exploited', 'stepped'],                           default=defaults.get('ucb_incr_mode', 'exploited'))
        parser.add_argument('--ucb-reset-mode',          type=str.lower, choices=['exploited', 'explored', 'explored_or_exploited'], default=defaults.get('ucb_reset_mode', 'exploited'))
        parser.add_argument('--ucb-subset-mode',         type=str.lower, choices=['all', 'exclude_bottom', 'top'],                   default=defaults.get('ucb_subset_mode', 'all'))
        parser.add_argument('--ucb-normalise-mode',      type=str.lower, choices=['population', 'subset'],                           default=defaults.get('ucb_normalise_mode', 'population'))
        parser.add_argument('--ucb-c',                   type=argparse_number_range(0, 2, float),                                    default=defaults.get('ucb_c', 0.1))
        # EXPLOITER - UCB & TS
        parser.add_argument('--ts-ratio-top',            type=argparse_number_range(1, float('inf'), int),                           default=defaults.get('ts_ratio_top', 0.2))
        parser.add_argument('--ts-ratio-bottom',         type=argparse_number_range(1, float('inf'), int),                           default=defaults.get('ts_ratio_bottom', 0.2))
        # EXTRA
        parser.add_argument('--debug',                   action='store_true')
        parser.add_argument('--enable-comet',            action='store_false', dest='disable_comet')

        return parser

    @staticmethod
    def _from_args(args: argparse.Namespace):
        return UcbExperimentArgs._from_dict(vars(args))

    @staticmethod
    def _from_dict(args) -> 'UcbExperimentArgs':
        return UcbExperimentArgs(*[args[field] for field in UcbExperimentArgs._fields])

    @staticmethod
    def from_system(defaults=None) -> 'UcbExperimentArgs':
        parser = UcbExperimentArgs._create_parser(defaults=defaults)
        return UcbExperimentArgs._from_args(parser.parse_args())


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
