# improving-pbt
My research on improving Population Based Training.


## My Notes:


### Slurm Commands:

```shell script
squeue                       # jobs submitted to the cluster queue
sinfo                        # state of the cluster
sinfo - R                    # state of the cluster with reasons for drained (disabled) nodes

salloc -N1 -p <queue>        # interactive session | N: num nodes | p: which cluster queue to use
srun -N1 -n1 -l <executable> # run job | N: num nodes | n: num tasks | l: label output per task

scancel <jobid>              # cancel specific job
scancel --user=<username>    # cancel all my jobs
```


### [Slurm Ray Submission Script](https://ray.readthedocs.io/en/latest/deploying-on-slurm.html):

```shell script
#!/bin/bash

#SBATCH --job-name=test
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=1GB
#SBATCH --nodes=5
#SBATCH --tasks-per-node 1

RAY_WORKERS=4 # One less that the total number of nodes
RAY_PORT=6379
RAW_WAIT=3

# ACTIVATE PYTHON
module load Langs/Python/3.6.4 # This will vary depending on your environment
source venv/bin/activate

# GET NODES
nodes_array=( $(scontrol show hostnames $SLURM_JOB_NODELIST) )
node_head="${nodes_array[0]}"

# GET RAY_ADDRESS & EXPORT
RAY_ADDRESS="$(srun --nodes=1 --ntasks=1 -w "${node_head}" hostname --ip-address):${RAY_PORT}"
export RAY_ADDRESS

# LAUNCH RAY - HEAD
srun --nodes=1 --ntasks=1 -w "${node_head}" ray start --block --head --redis-port=6379 &
sleep "${RAW_WAIT}"

# LAUNCH RAY - NODES
for ((  i=1; i<=${RAY_WORKERS}; i++ )); do
  node_worker=${nodes_array[$i]}
  srun --nodes=1 --ntasks=1 -w "${node_worker}" ray start --block --address=$ip_head &
done
sleep 5

# START RAY SCRIPT
python trainer.py 100 # Pass the total number of allocated CPUs
```

main.py
```python
import os
import ray

ray.init(address=os.environ["RAY_ADDRESS"])

# TODO: REST OF SCRIPT
```



### Development Environment:

- [Install Pyenv](https://github.com/pyenv/pyenv#installation):

  See the instructions [here](https://github.com/pyenv/pyenv#installation)


- Install miniconda3/python3.7:

  ```shell script
  pyenv install miniconda3-latest
  # Activate globally
  pyenv global miniconda3-latest
  ```


- [Enlarge Git Pull Buffer](https://stackoverflow.com/questions/38378914/git-error-rpc-failed-curl-56-gnutls) (If installing PyTorch from source):

  ```shell script
  git config --global http.postBuffer 1048576000
  ````


- [Install PyTorch](https://pytorch.org/get-started/locally):

  ```shell script
  conda install opencv                   # optional: opencv support
  conda install -c conda-forge openmpi   # optional: openmpi support

  conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
  ```

- [Install Ray 0.8.0](https://ray.readthedocs.io/en/latest/installation.html#latest-snapshots-nightlies):

  ```shell script
  pip install mujoco-py  # optional: mujoco physics simulations for openai gym (requires local license)
  
  pip install pandas                     # optional: ray extra, analysis utilities
  pip install psutil                     # optional: ray extra, memory usage
  pip install setproctitle               # optional: ray extra, custom worker names
  pip install tensorflow-gpu==2.0.0-beta # optional: stop annying ray warnings
  
  pip install <ray-0.8.0 wheel url>      # https://ray.readthedocs.io/en/latest/installation.html#latest-snapshots-nightlies
  ```

    



    
    
    

