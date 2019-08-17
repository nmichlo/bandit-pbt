#!/bin/bash
# Based on: https://github.com/ray-project/ray/issues/826

# ------------------------------------------------------------------------- #
# UTIL                                                                      #
# ------------------------------------------------------------------------- #

function default { if [ -z "${!1}" ]; then declare "$1"="$2" ; export "${1?}" ; fi ; printf "\033[90m${1}\033[0m=\"\033[32m${!1}\033[0m\" (\033[95moptional\033[0m)\n" ; }
function defined { if [ -z "${!1}" ]; then printf "\033[91mInput Error: \033[90m$1\033[91m is not defined... EXITING\033[0m!\n" ; exit 1 ; fi ; printf "\033[90m${1}\033[0m=\"\033[32m${!1}\033[0m\" (\033[94minfo\033[0m)\n" ; }
function required { if [ -z "${!1}" ]; then printf "\033[91mRuntime Error: \033[90m$1\033[91m is not defined... EXITING!\033[0m\n" ; exit 1 ; fi ; printf "\033[90m${1}\033[0m=\"\033[32m${!1}\033[0m\" (\033[93mrequired\033[0m)\n" ; }

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# ------------------------------------------------------------------------- #
# VARS                                                                      #
# ------------------------------------------------------------------------- #

defined SCRIPT_DIR

default RAY_SCRIPT_START_HEAD   "$SCRIPT_DIR/start_head.sh"
default RAY_SCRIPT_START_WORKER "$SCRIPT_DIR/start_worker.sh"
default RAY_REDIS_PORT          6379
default RAY_STORAGE_FOLDER      "./pid_storage"
default RAY_WAIT                3

required RAY_PY_MAIN

# ------------------------------------------------------------------------- #
# SLURM INFO                                                                #
# ------------------------------------------------------------------------- #

defined SLURM_JOB_NODELIST

RAY_NODES=($(scontrol show hostnames $SLURM_JOB_NODELIST))
export RAY_HEAD_IP="$(srun --nodes=1 --ntasks=1 -w "${RAY_NODES[0]}" hostname --ip-address):$RAY_REDIS_PORT"

defined RAY_NODES
defined RAY_HEAD_IP

# ------------------------------------------------------------------------- #
# START                                                                     #
# ------------------------------------------------------------------------- #

srun --nodes=1 --ntasks=1 -w "${RAY_NODES[0]}" "$RAY_SCRIPT_START_HEAD" "$RAY_REDIS_PORT" "$RAY_STORAGE_FOLDER" &
sleep "$RAY_WAIT"
for ((  i=1; i < "${#RAY_NODES[@]}"; i++ )); do
     srun --nodes=1 --ntasks=1 -w "${RAY_NODES[$i]}" "$RAY_SCRIPT_START_WORKER" "$RAY_HEAD_IP" "$RAY_STORAGE_FOLDER" "$i" &
     sleep "$RAY_WAIT"
done

# ------------------------------------------------------------------------- #
# RUN                                                                       #
# ------------------------------------------------------------------------- #

python "$RAY_PY_MAIN"

# ------------------------------------------------------------------------- #
# QUIT                                                                      #
# ------------------------------------------------------------------------- #

pkill -P "$(cat "$RAY_STORAGE_FOLDER/head.pid")" sleep
for ((  i=1; i < "${#RAY_NODES[@]}"; i++ )); do
    pkill -P "$(cat "$RAY_STORAGE_FOLDER/worker${i}.pid")" sleep
done