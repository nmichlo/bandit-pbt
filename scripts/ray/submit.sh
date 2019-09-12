
# ------------------------------------------------------------------------- #
# UTIL                                                                      #
# ------------------------------------------------------------------------- #

function default { if [ -z "${!1}" ]; then declare "$1"="$2" ; export "${1?}" ; fi ; printf "\033[90m${1}\033[0m=\"\033[32m${!1}\033[0m\" (\033[95moptional\033[0m)\n" ; }
function defined { if [ -z "${!1}" ]; then printf "\033[91mRuntime Error: \033[90m$1\033[91m is not defined... EXITING\033[0m!\n" ; exit 1 ; fi ; printf "\033[90m${1}\033[0m=\"\033[32m${!1}\033[0m\" (\033[94minfo\033[0m)\n" ; }
function required { if [ -z "${!1}" ]; then printf "\033[91mInput Error: \033[90m$1\033[91m is not defined... EXITING!\033[0m\n" ; exit 1 ; fi ; printf "\033[90m${1}\033[0m=\"\033[32m${!1}\033[0m\" (\033[93mrequired\033[0m)\n" ; }

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# ------------------------------------------------------------------------- #
# VARS                                                                      #
# ------------------------------------------------------------------------- #

default RAY_SCRIPT "$@"

default RAY_WORKERS $(("$SLURM_JOB_NUM_NODES" - 1))
default RAY_PORT 6379
default RAY_WAIT 3

defined SLURM_JOB_NAME
defined SLURM_JOB_ID
defined SLURM_JOB_NUM_NODES
defined SLURM_SUBMIT_DIR
defined SLURM_SUBMIT_HOST

# ------------------------------------------------------------------------- #
# SLURM INFO                                                                #
# ------------------------------------------------------------------------- #

# ACTIVATE PYTHON
module load Langs/Python/3.6.4 # This will vary depending on your environment
source venv/bin/activate

# GET NODES
nodes_array=( $(scontrol show hostnames $SLURM_JOB_NODELIST) )
node_head="${nodes_array[0]}"

# GET RAY_ADDRESS & EXPORT
RAY_ADDRESS="$(srun --nodes=1 --ntasks=1 -w "${node_head}" hostname --ip-address):${RAY_PORT}"
export RAY_ADDRESS

# ------------------------------------------------------------------------- #
# SLURM INFO                                                                #
# ------------------------------------------------------------------------- #

# LAUNCH RAY - HEAD
srun --nodes=1 --ntasks=1 -w "${node_head}" ray start --block --head --redis-port=6379 &
sleep "${RAY_WAIT}"

# LAUNCH RAY - NODES
for ((  i=1; i<=${RAY_WORKERS}; i++ )); do
  node_worker=${nodes_array[$i]}
  srun --nodes=1 --ntasks=1 -w "${node_worker}" ray start --block --address="${RAY_ADDRESS}" &
done
sleep "${RAY_WAIT}"

# ------------------------------------------------------------------------- #
# RUN                                                                       #
# ------------------------------------------------------------------------- #

# START RAY SCRIPT
python "${RAY_SCRIPT}" | tee "${SLURM_JOB_ID}_log.txt"
