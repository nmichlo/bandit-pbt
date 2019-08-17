#!/bin/bash
# Based on: https://github.com/ray-project/ray/issues/826

if [ -z "$1" ]; then echo "[arg: 1] Undefined: RAY_HEAD_ADDRESS" ; fi
if [ -z "$2" ]; then echo "[arg: 2] Undefined: RAY_STORAGE_FOLDER" ; fi
if [ -z "$3" ]; then echo "[arg: 3] Undefined: RAY_WORKER_ID" ; fi

# LAUNCH
ray start --redis-address="$1"
echo "$$" | tee "$2/worker$3.pid"
sleep infinity