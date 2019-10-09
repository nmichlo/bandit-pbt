
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

def seed(seed=42):
    # https://pytorch.org/docs/stable/notes/randomness.html
    import random
    import numpy as np
    import torch.backends.cudnn
    # python
    random.seed(seed)
    # numpy
    np.random.seed(seed)
    # torch
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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


def shuffled(x, enabled=True):
    items = list(x)
    if enabled:
        import random
        random.shuffle(items)
    return items


def sorted_random_ties(a, key=None):
    import random
    if key is None:
        return sorted(a, key=lambda x: (x, random.random()))
    else:
        return sorted(a, key=lambda x: (key(x), random.random()))

def argsorted_random_ties(a, key=None):
    indices = range(len(a))
    if key is None:
        return sorted_random_ties(indices, key=lambda i: a[i])
    else:
        return sorted_random_ties(indices, key=lambda i: key(a[i]))

def make_empty_dir(path):
    import os
    # make sure the folder is empty and exists
    os.makedirs(path, exist_ok=True)
    for file in os.listdir(path):
        os.remove(os.path.join(path, file))
    return path


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
    from tqdm import tqdm
    # get classes
    classes = get_module_classes(module)
    parent_childrens = {}
    for name, cls in classes.items():
        (parent,) = cls.__bases__
        parent_childrens.setdefault(parent.__name__, []).append(name)
    def recurse(name, depth=0):
        tqdm.write('    ' * depth, '-', f'{name:20s}', inspect.signature(classes[name]) if name in classes else '')
        if name in parent_childrens:
            for k in parent_childrens[name]:
                recurse(k, depth+1)
    recurse(root_cls_name)


# ========================================================================= #
# STRINGS                                                                   #
# ========================================================================= #


def print_separator(text, width=100, char_v='#', char_h='=', char_corners=None):
    """
    function wraps text in a ascii box.
    """
    if char_corners is None:
        char_corners = char_v
    assert len(char_v) == len(char_corners)
    assert len(char_h) == 1
    import textwrap
    import pprint
    from tqdm import tqdm

    w = width-4
    lines = []
    lines.append(f'\n{char_corners} {char_h*w} {char_corners}')
    if type(text) != str:
        text = pprint.pformat(text, width=w)
    for line in text.splitlines():
        for wrapped in (textwrap.wrap(line, w, tabsize=4) if line.strip() else ['']):
            lines.append(f'{char_v} {wrapped:{w}s} {char_v}')
    lines.append(f'{char_corners} {char_h*w} {char_corners}\n')
    tqdm.write('\n'.join(lines))


# ========================================================================= #
# TIMINGS                                                                   #
# ========================================================================= #


def min_time_elapsed(func_or_seconds, seconds=None):
    """
    Decorator that only runs a function if the minimum time has elapsed since the last run.
    """
    import time
    last_time = None
    def decorator(func):
        def inner(*args, **kwargs):
            nonlocal last_time
            curr_time = time.time()
            if last_time is None:  # wait until <seconds> after first call
                last_time = curr_time
            if last_time + seconds <= curr_time:
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
# DOTENV                                                                    #
# ========================================================================= #


def load_dotenv():
    import dotenv
    env_path = dotenv.find_dotenv(raise_error_if_not_found=True)
    dotenv.load_dotenv(env_path, verbose=True)
    # LOAD ENVIRONMENT VALUES
    values_all = dotenv.dotenv_values(env_path, verbose=True)
    values = {k: v for k, v in values_all.items() if not (k.lower().startswith('http') or 'key' in k.lower())}
    # PRINT
    string = '\n'.join(f'{k}: {v}' for k, v in values.items())
    print_separator(f'[LOADED ENVIRONMENT]: {env_path}\n[HIDDEN KEYS]: {", ".join(set(values_all)-set(values))}\n\n{string}')
    return values


def get_python_path():
    import os
    return os.environ.get('PYTHONPATH', '').split(os.pathsep)

def strip_python_path(path):
    paths = get_python_path()
    paths = [path[len(p):].lstrip('\\/') for p in paths if path.startswith(p)]
    if not paths:
        return path
    return min(paths, key=len)

def strip_pwd(path):
    import os
    if 'PWD' not in os.environ:
        return path
    p = os.environ.get('PWD')
    if path.startswith(p):
        return path[len(p):].lstrip('\\/')
    return path

def simplify_path(path, strip_pwd_=False, strip_python_path_=False):
    assert not (strip_pwd_ and strip_python_path_), 'Choose either strip_pwd or strip_python_path, cannot do both.'
    if strip_python_path_:
        path = strip_python_path(path)
    if strip_pwd_:
        path = strip_pwd(path)
    return path

# ========================================================================= #
# END                                                                       #
# ========================================================================= #


