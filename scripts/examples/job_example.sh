#SBATCH --job-name=ray_pytorch_mnist
#SBATCH --nodes=5
#SBATCH --tasks-per-node 1

# get the location of the current directory
SCRIPT_DIR="$(cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd)"

# brinup ray head and workers, defines $RAY_ADDRESS
source "${SCRIPT_DIR}/ray/bringup_ray_cluster.sh"

# launch ray script
python3 -m ray.tune.examples.mnist_pytorch_trainable --ray-address "${RAY_ADDRESS}"