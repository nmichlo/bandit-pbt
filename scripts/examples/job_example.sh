#SBATCH --job-name=ray_pytorch_mnist
#SBATCH --nodes=5
#SBATCH --tasks-per-node 1

# get the location of the current directory
SCRIPT_DIR="$(cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd)"

# brinup ray head and workers, defines $RAY_ADDRESS & stop_ray
source "$(dirname "${SCRIPT_DIR}")/bringup_ray_cluster.sh"

stop_ray
start_ray

## launch ray script
srun --nodes=1 --ntasks=1 -w "$RAY_HEAD_NODE" \
  python3 -m ray.tune.examples.mnist_pytorch_trainable --ray-address "${RAY_ADDRESS}" \
  | tee "job-${SLURM_JOB_ID}.log"

# stop ray head and workers
stop_ray