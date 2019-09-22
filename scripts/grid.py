
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
from pathlib import Path
from string import Formatter
import sys


# ========================================================================= #
# HELPER                                                                    #
# ========================================================================= #


def assertion(bool, message):
    if not bool:
        sys.stderr.write(f'\033[91m[ERROR]: {message}\033[0m\n')
        exit(1)


def is_int(val):
    try:
        int(val)
        return True
    except:
        return False


def grid_search(choices):
    def recurse(choices):
        if len(choices) == 1:
            for val in choices[0]:
                yield (val,)
        else:
            for val in choices[0]:
                for values in recurse(choices[1:]):
                    yield (val, *values)
    for values in recurse(list(choices.values())):
        yield {k: v for k, v in zip(choices, values)}


# ========================================================================= #
# MAIN                                                                      #
# ========================================================================= #


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('template', type=str)
    parser.add_argument('-c', '--choices', action='append', metavar=('key', 'values'), nargs='+', default=[])
    parser.add_argument('-o', '--out', type=Path, default=None)
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-p', '--print', action='store_true')

    args = parser.parse_args()

    # check choices
    assertion(all(len(c) >= 2 for c in args.choices), 'Each --choices must have a key an value eg. -c key: value')
    assertion(all(len(c[0]) >= 2 for c in args.choices), 'Each key must be at least one character long')
    assertion(all(c[0][-1] == ':' for c in args.choices), 'Each choice must have a key ending in ":" eg. -c names: bob ken')
    assertion(len(set(c[0] for c in args.choices)) == len(args.choices), 'Each --choices key must be unique')
    assertion(all(' ' not in c[0] for c in args.choices), 'Each choice key cannot contain spaces')

    choices = {c[0][:-1]: c[1:] for c in args.choices}
    assertion(all(not is_int(k) for k in choices.keys()), 'Choice keys cannot be positional (integers)')

    # check fields
    fields = list(field_name.strip() for (_, field_name, _, _) in Formatter().parse(args.template) if field_name is not None)
    assertion(all(len(field) >= 1 for field in fields), 'All fields of the template must be named')
    assertion(all(' ' not in field for field in fields), 'Template field names cannot contain spaces')
    assertion(all(not is_int(field) for field in fields), 'Template field names cannot be positional (integers)')

    # check missing
    assertion(len(set(choices) - set(fields)) == 0, 'A key from the choices does not belong to the named template fields')
    assertion(len(set(fields) - set(choices)) == 0, 'A named field from the template does not belong to the choices keys')

    # PRINT
    if args.verbose:
        print('\n[OPTIONS]:')
        max_len = max(len(k) for k in choices)
        for k, v in choices.items():
            print(f'    {k:{max_len}s}: {v}')

    # GRID SEARCH
    results = [args.template.format_map(chosen) for chosen in grid_search(choices)]

    if args.verbose:
        print(f'[PERMUTATIONS]: {len(results)}')

    # SAVE
    if args.out:
        with open(args.out, 'a') as file:
            for line in results:
                file.write(line)
                file.write('\n')
        if args.verbose:
            print(f'[SAVED]: {args.out}')


    # PRINT
    if not args.out or args.print:
        if args.verbose:
            print('\n[GRID SEARCH]:')
        for result in results:
            print(result)


# ========================================================================= \#
# END                                                                       \#
# ========================================================================= \#
