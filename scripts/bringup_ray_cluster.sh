#!/bin/bash

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
# SLURM CLUSTER RAY BRINGUP                                                 #
#                                                                           #
# This file is intended to be sourced by a slurm job script. Which then     #
# makes use of the `ray_start` and `ray_stop` functions to surrounding the  #
# contents of the script to binging up and shutting down the ray cluster.   #
#                                                                           #
# It is alright to call `ray_stop` before `ray_start`, thus twice in the    #
# script to be on the safe side.                                            #
#                                                                           #
# The reason this script exists is so that ray can be used within an        #
# interactive slurm job created via `salloc`.                               #
# ========================================================================= #

# ------------------------------------------------------------------------- #
# UTIL                                                                      #
# ------------------------------------------------------------------------- #

function default { if [ -z "${!1}" ]; then readonly "$1"="$2" ; fi ; printf "\033[90m${1}\033[0m=\"\033[32m${!1}\033[0m\" (\033[95moptional\033[0m)\n" ; }
function defined { if [ -z "${!1}" ]; then printf "\033[91mError: variable \033[90m$1\033[91m is not defined... EXITING\033[0m!\n" ; exit 1 ; fi ; printf "\033[90m${1}\033[0m=\"\033[32m${!1}\033[0m\" (\033[94minfo\033[0m)\n" ; }
function exists  { if [ -z "$(command -v "$1")" ]; then printf "\033[91mError: command \033[90m$1\033[91m not found... EXITING\033[0m!\n" ; exit 1 ; fi ; }

# ------------------------------------------------------------------------- #
# VARS                                                                      #
# ------------------------------------------------------------------------- #

exists srun
exists scontrol

default RAY_NUM_WORKERS $(("$SLURM_JOB_NUM_NODES" - 1))
default RAY_PORT 6379
default RAY_WAIT 5

echo

defined SLURM_JOB_NAME
defined SLURM_JOB_ID
defined SLURM_JOB_NUM_NODES
defined SLURM_SUBMIT_DIR
defined SLURM_SUBMIT_HOST
defined SLURM_JOB_NODELIST

echo

# ------------------------------------------------------------------------- #
# SLURM INFO                                                                #
# ------------------------------------------------------------------------- #

# GET NODES
RAY_NODES=( $(scontrol show hostnames $SLURM_JOB_NODELIST) )
RAY_HEAD_NODE="${RAY_NODES[0]}"

# GET RAY_ADDRESS & EXPORT
RAY_ADDRESS="$(srun --nodes=1 --ntasks=1 -w "${RAY_NODES[0]}" hostname --ip-address):${RAY_PORT}"

# ------------------------------------------------------------------------- #
# SLURM INFO                                                                #
# ------------------------------------------------------------------------- #

function start_ray() {
    # LAUNCH RAY - HEAD
    echo "[RAY]: LAUNCHING HEAD"
    srun --nodes=1 --ntasks=1 -w "${RAY_NODES[0]}" ray start --block --head --redis-port="${RAY_PORT}" &
    sleep "${RAY_WAIT}"

    # LAUNCH RAY - NODES
    echo "[RAY]: LAUNCHING WORKERS"
    for ((  i=1; i<=$RAY_NUM_WORKERS; i++ )); do
      srun --nodes=1 --ntasks=1 -w "${RAY_NODES[$i]}" ray start --block --address="${RAY_ADDRESS}" &
    done
    sleep "${RAY_WAIT}"
}

# ------------------------------------------------------------------------- #
# STOP                                                                      #
# ------------------------------------------------------------------------- #

function stop_ray() {
    # STOP RAY - NODES
    echo "[RAY]: STOPPING WORKERS (Errors here are expected)"
    for ((  i=1; i<=$RAY_NUM_WORKERS; i++ )); do
      srun --nodes=1 --ntasks=1 -w "${RAY_NODES[$i]}" ray stop &
    done
    sleep "${RAY_WAIT}"

    # STOP RAY - HEAD
    echo "[RAY]: STOPPING HEAD (Errors here are expected)"
    srun --nodes=1 --ntasks=1 -w "${RAY_NODES[0]}" ray stop &
    sleep "${RAY_WAIT}"
}

# ------------------------------------------------------------------------- #
# END                                                                       #
# ------------------------------------------------------------------------- #
