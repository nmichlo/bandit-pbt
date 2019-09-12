
# ------------------------------------------------------------------------- #
# UTIL                                                                      #
# ------------------------------------------------------------------------- #

function default { if [ -z "${!1}" ]; then declare "$1"="$2" ; export "${1?}" ; fi ; printf "\033[90m${1}\033[0m=\"\033[32m${!1}\033[0m\" (\033[95moptional\033[0m)\n" ; }
function defined { if [ -z "${!1}" ]; then printf "\033[91mError: variable \033[90m$1\033[91m is not defined... EXITING\033[0m!\n" ; exit 1 ; fi ; printf "\033[90m${1}\033[0m=\"\033[32m${!1}\033[0m\" (\033[94minfo\033[0m)\n" ; }
function exists  { if [ -z "$(command -v "${!1}")" ]; then printf "\033[91mError: command \033[90m$1\033[91m not found... EXITING\033[0m!\n" ; exit 1 ; fi ; }

# ------------------------------------------------------------------------- #
# VARS                                                                      #
# ------------------------------------------------------------------------- #

default RAY_WORKERS $(("$SLURM_JOB_NUM_NODES" - 1))
default RAY_PORT 6379
default RAY_WAIT 3

exists srun
exists scontrol

defined SLURM_JOB_NAME
defined SLURM_JOB_ID
defined SLURM_JOB_NUM_NODES
defined SLURM_SUBMIT_DIR
defined SLURM_SUBMIT_HOST

# ------------------------------------------------------------------------- #
# SLURM INFO                                                                #
# ------------------------------------------------------------------------- #

# GET NODES
_nodes=( $(scontrol show hostnames $SLURM_JOB_NODELIST) )

# GET RAY_ADDRESS & EXPORT
_ray_address="$(srun --nodes=1 --ntasks=1 -w "${nodes[0]}" hostname --ip-address)"
defined _ray_address

RAY_ADDRESS="${_ray_address}:${RAY_PORT}"
export RAY_ADDRESS

# ------------------------------------------------------------------------- #
# SLURM INFO                                                                #
# ------------------------------------------------------------------------- #

# LAUNCH RAY - HEAD
echo "[RAY]: LAUNCHING HEAD"
srun --_nodes=1 --ntasks=1 -w "${_nodes[0]}" ray start --block --head --redis-port=6379 &
sleep "${RAY_WAIT}"

# LAUNCH RAY - NODES
echo "[RAY]: LAUNCHING WORKERS"
for ((  i=1; i<=$RAY_WORKERS; i++ )); do
  srun --_nodes=1 --ntasks=1 -w "${_nodes[$i]}" ray start --block --address="${RAY_ADDRESS}" &
done
sleep "${RAY_WAIT}"

# ------------------------------------------------------------------------- #
# CLEANUP                                                                   #
# ------------------------------------------------------------------------- #

unset default
unset defined
unset exists

unset RAY_WORKERS
unset RAY_PORT
unset RAY_WAIT

unset _nodes
unset _ray_address