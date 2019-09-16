
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

# ========================================================================= #
# util                                                                   #
# ========================================================================= #

def is_iterable(obj):
    if type(obj) in {str}:
        return False
    try:
        for _ in obj:
            return True
    except:
        return False

def grid_search(choices_list):
    assert type(choices_list) != dict
    if len(choices_list) == 1:
        for choice in choices_list[0]:
            yield (choice,)
    elif len(choices_list) > 1:
        for choice in choices_list[0]:
            for combined in grid_search(choices_list[1:]):
                yield (choice, *combined)

def grid_search_named(choices_dict, defaults=None, disabled=None):
    disabled = set(disabled or {})
    iterable_keys = [k for k, v in choices_dict.items() if is_iterable(v) and k not in disabled]
    # defaults not in choices dict, supports iterables like lists and stuff
    defaults = {**(defaults or {}), **{k: v for k, v in choices_dict.items() if not is_iterable(v) or k in disabled}}
    for choices in grid_search([choices_dict[k] for k in iterable_keys]):
        choices = {k: v for k, v in zip(iterable_keys, choices)}
        if defaults:
            choices = {**defaults, **choices}  # named_choices takes priority
        yield choices

def confidence_interval(data, confidence=0.95):
    from scipy.stats import sem, t
    from scipy import mean
    import numpy as np
    # calculate
    if len(data) < 2:
        return np.array(np.nan)
    h = sem(data) * t.ppf((1 + confidence) / 2, len(data) - 1)
    return h  # - np.array([-h, +h]) # + mean(data)



# ========================================================================= \#
# END                                                                       \#
# ========================================================================= \#
