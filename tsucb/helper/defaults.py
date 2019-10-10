
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
# defaults                                                                   #
# ========================================================================= #


def set_defaults(seaborn=True, pandas=True):
    # DEFAULTS - SEABORN
    if seaborn:
        import seaborn as sns
        sns.set(
            context='paper',
            style='whitegrid',
            font='serif',
            font_scale=1.25,
            palette='GnBu_d',
            rc={
                'grid.linestyle': ':',
                'grid.color': '.9',
                'axes.edgecolor': '.1',
                'text.color': '.1',
                # 'font.serif': ['Times New Roman']
            },
        )

    # DEFAULTS - PANDAS
    if pandas:
        import pandas as pd
        pd.set_option('display.max_rows', 50)
        pd.set_option('display.max_columns', 10)
        pd.set_option('display.width', 100)


# ========================================================================= \#
# END                                                                       \#
# ========================================================================= \#
