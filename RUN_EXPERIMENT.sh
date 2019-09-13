#SBATCH --job-name=ray_pbt
#SBATCH --tasks-per-node 1

# get the location of the current directory
SCRIPT_DIR="$(cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd)/scripts"
# bringup ray head and workers, defines $RAY_ADDRESS & stop_ray
source "${SCRIPT_DIR}/bringup_ray_cluster.sh"

# START
stop_ray
start_ray

# RUN
srun --nodes=1 --ntasks=1 -w "$RAY_HEAD_NODE" \
  python3 "src/experiment.py" \
  | tee "job-${SLURM_JOB_ID}.log"

# STOP
stop_ray