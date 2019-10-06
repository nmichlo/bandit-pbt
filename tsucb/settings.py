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


import logging
import multiprocessing

import dotenv
import os
import torch


# ========================================================================= #
# .env                                                                     #
# ========================================================================= #
from tqdm import tqdm

dotenv.load_dotenv(dotenv.find_dotenv(), verbose=True)


# ========================================================================= #
# LOGGING                                                                   #
# ========================================================================= #


_LOG_LEVELS = {
    'CRITICAL': logging.CRITICAL,  # = 50
    'FATAL':    logging.FATAL,     # = CRITICAL
    'ERROR':    logging.ERROR,     # = 40
    'WARNING':  logging.WARNING,   # = 30
    'WARN':     logging.WARN,      # = WARNING
    'INFO':     logging.INFO,      # = 20
    'DEBUG':    logging.DEBUG,     # = 10
    'NOTSET':   logging.NOTSET,    # = 0
}

LOG_LEVEL = os.getenv('LOG_LEVEL', 'ERROR')
assert LOG_LEVEL in _LOG_LEVELS


# ========================================================================= #
# COMET.ML                                                                  #
# ========================================================================= #


# https://www.comet.ml/docs/python-sdk/advanced/#comet-configuration-variables

# Configuration item     | Section/Name                   | Environment variable         | Description
# -----------------------+--------------------------------+------------------------------+------------
# API key                | [comet] api_key                | COMET_API_KEY                | The API key used for creating an Experiment
# REST API key           | [comet] rest_api_key           | COMET_REST_API_KEY           | The REST API key
# Current Workspace      | [comet] workspace              | COMET_WORKSPACE              | Use this workspace when creating new experiments
# Current Project        | [comet] project_name           | COMET_PROJECT_NAME           | Use this project when creating new experiments
# Logging file           | [comet] logging_file           | COMET_LOGGING_FILE           | Use the given file for storing log messages. This can also contain patterns such as "comet-{project}.log"
# Logging file level     | [comet] logging_file_level     | COMET_LOGGING_FILE_LEVEL     | By default, the log outputs will contains INFO and higher level (WARNING and ERROR) log messages. This configuration item can be used to change the level of the logged messages.
# Logging file overwrite | [comet] logging_file_overwrite | COMET_LOGGING_FILE_OVERWRITE | Overwrites the log file on each run, if True
# Console logging        | [comet] console                | COMET_CONSOLE                | Set the logging level for console messages (e.g., INFO, DEBUG)
# Offline Directory      | [comet] offline_directory      | COMET_OFFLINE_DIRECTORY      | Set the offline directory for OfflineExperiment()
# Display Summary        | [comet] display_summary        | COMET_DISPLAY_SUMMARY        | If False, do not display upload summary for experiments


ENABLE_COMET_ML = bool(os.getenv('ENABLE_COMET_ML'))
ENABLE_COMET_ML = ENABLE_COMET_ML and (os.getenv('COMET_API_KEY') and os.getenv('COMET_PROJECT_NAME') and os.getenv('COMET_WORKSPACE'))

if not ENABLE_COMET_ML:
    tqdm.write('\033[91mWARNING: Comet.ml Experiment Tracking Disabled\033[0m')


# ========================================================================= #
# RAY                                                                       #
# ========================================================================= #


RAY_ADDRESS = os.getenv('RAY_ADDRESS', None)


# ========================================================================= #
# RESOURCES                                                                 #
# ========================================================================= #


USE_GPU = bool(os.getenv('USE_GPU', torch.cuda.is_available())) and torch.cuda.is_available()
CPUS_PER_NODE = int(os.getenv('CPUS_PER_NODE', multiprocessing.cpu_count()))


# ========================================================================= #
# EXPERIMENT                                                                #
# ========================================================================= #


EXP_POPULATION_SIZE = os.getenv('POPULATION_SIZE', 4)
EXP_SCHEDULER = os.getenv('SCHEDULER', 'pbt')


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
