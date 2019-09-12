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

job.sh
```shell script
#!/bin/bash

#SBATCH --job-name=ray_pytorch_mnist
#SBATCH --nodes=5
#SBATCH --tasks-per-node 1

# bringup ray head and workers, defines $RAY_ADDRESS
source ./scripts/ray/bringup_ray_cluster.sh
# launch ray script
python main.py | tee "${SLURM_JOB_ID}_log.txt"
```


main.py
```python
import os
import ray

ray.init(address=os.environ["RAY_ADDRESS"])
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

    



    
    
    

