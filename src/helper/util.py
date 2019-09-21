
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
        return np.array(0)
    h = sem(data) * t.ppf((1 + confidence) / 2, len(data) - 1)
    return h  # - np.array([-h, +h]) # + mean(data)


# ========================================================================= #
# MODULES                                                                   #
# ========================================================================= #


def get_module_objects(module):
    names = [name for name in dir(module) if not name.startswith('__')]
    return {name: getattr(module, name) for name in names}

def get_module_type(module, type_name):
    objects = get_module_objects(module)
    return {name: obj for name, obj in objects.items() if type(obj).__name__ == type_name}

def get_module_classes(module, filter_nonlocal=False):
    classes = get_module_type(module, 'type')
    if filter_nonlocal:
        classes = {name: cls for name, cls in get_module_type(module, 'type').items() if cls.__module__ == module.__name__}
    return classes
def get_module_functions(module): return get_module_type(module, 'function')
def get_module_modules(module): return get_module_type(module, 'module')

def print_module_class_heirarchy(module, root_cls_name):
    import inspect
    # get classes
    classes = get_module_classes(module)
    parent_childrens = {}
    for name, cls in classes.items():
        (parent,) = cls.__bases__
        parent_childrens.setdefault(parent.__name__, []).append(name)
    def recurse(name, depth=0):
        print('    ' * depth, '-', f'{name:20s}', inspect.signature(classes[name]) if name in classes else '')
        if name in parent_childrens:
            for k in parent_childrens[name]:
                recurse(k, depth+1)
    recurse(root_cls_name)


# ========================================================================= #
# STRINGS                                                                   #
# ========================================================================= #


def print_separator(text, width=100):
    """
    function wraps text in a ascii box.
    """
    import textwrap
    import pprint
    w = width-4
    lines = []
    lines.append(f'\n# {"="*w} #')
    if type(text) != str:
        text = pprint.pformat(text, width=w)
    for line in text.splitlines():
        for wrapped in textwrap.wrap(line, w, tabsize=4):
            lines.append(f'# {wrapped:{w}s} #')
    lines.append(f'# {"="*w} #\n')
    print('\n'.join(lines))


# ========================================================================= #
# TIMINGS                                                                   #
# ========================================================================= #


def min_time_elapsed(func_or_seconds, seconds=None):
    """
    Decorator that only runs a function if the minimum time has elapsed since the last run.
    """
    import time
    last_time = 0
    def decorator(func):
        def inner(*args, **kwargs):
            nonlocal last_time
            curr_time = time.time()
            if last_time + seconds >= curr_time:
                last_time = curr_time
                return func(*args, **kwargs)
            return None
        return inner
    if callable(func_or_seconds):
        assert isinstance(seconds, (int, float))
        return decorator(func_or_seconds)
    else:
        assert isinstance(func_or_seconds, (int, float))
        seconds = func_or_seconds
        return decorator


# ========================================================================= #
# END                                                                       #
# ========================================================================= #


