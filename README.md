# improving-pbt
My research on improving Population Based Training.


## My Notes:


### Slurm Commands:

```shell script
squeue                       # jobs submitted to the cluster queue
sinfo                        # state of the cluster
sinfo - R                    # state of the cluster with reasons for drained (disabled) nodes

salloc -N1 -p {queue}               # interactive session | N: num nodes | p: which cluster queue to use
sbatch -N1 -p {queue} {job_script}  # submit a job file   | N: num nodes | p: which cluster queue to use

srun -N1 -n1 -p {queue} -l {executable}   # run command   | N: num nodes | p: which cluster queue to use | n: num tasks | l: label output per task
                                          # used inside a job file or interactive session to make a new step

scancel {jobid{.step}}       # cancel specific job
scancel --user={username}    # cancel all my jobs
scancel $(squeue | grep {username} | awk '{print $1;}') # alternative cancel all jobs

sacct                        # view detailed job logs for your account
sacct -j {jobid}             # view all job steps

# cancel current steps of a job, useful in an salloc session
scancel $(sacct -j {jobid} | grep RUNNING | grep "{jobid}\." | awk '{print $1}')
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
  
  Fix conda ssl errors
  ```shell script
  conda install -c conda-forge certifi
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
