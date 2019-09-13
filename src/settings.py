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


import multiprocessing
import dotenv
import os
import torch


# ========================================================================= #
# .env                                                                     #
# ========================================================================= #


dotenv.load_dotenv(dotenv.find_dotenv(), verbose=True)


# ========================================================================= #
# COMET.ML                                                                  #
# ========================================================================= #


COMET_ML_API_KEY = os.getenv('COMET_ML_API_KEY')
COMET_ML_PROJECT_NAME = os.getenv('COMET_ML_PROJECT_NAME')
COMET_ML_WORKSPACE = os.getenv('COMET_ML_WORKSPACE')

ENABLE_COMET_ML = bool(os.getenv('ENABLE_COMET_ML'))
ENABLE_COMET_ML = ENABLE_COMET_ML and (COMET_ML_API_KEY and COMET_ML_PROJECT_NAME and COMET_ML_WORKSPACE)

if not ENABLE_COMET_ML:
    print('\033[91mWARNING: Comet.ml Experiment Tracking Disabled\033[0m')


# ========================================================================= #
# RAY                                                                       #
# ========================================================================= #


RAY_ADDRESS = os.getenv('RAY_ADDRESS')


# ========================================================================= #
# RESOURCES                                                                 #
# ========================================================================= #


USE_GPU = bool(os.getenv('USE_GPU', torch.cuda.is_available())) and torch.cuda.is_available()
CPUS_PER_NODE = int(os.getenv('USE_GPU', multiprocessing.cpu_count()))


# ========================================================================= #
# EXPERIMENT                                                                #
# ========================================================================= #


POPULATION_SIZE = os.getenv('POPULATION_SIZE', 20)


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
