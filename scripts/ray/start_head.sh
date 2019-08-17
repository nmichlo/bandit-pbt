#!/bin/bash
# Based on: https://github.com/ray-project/ray/issues/826

if [ -z "$1" ]; then echo "[arg: 1] Undefined: RAY_REDIS_PORT" ; fi
if [ -z "$2" ]; then echo "[arg: 2] Undefined: RAY_STORAGE_FOLDER" ; fi

# LAUNCH
ray start --head --redis-port="$1"
echo "$$" | tee "$2/head.pid"
sleep infinity