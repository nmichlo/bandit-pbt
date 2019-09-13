#  Copyright (c) 2019 Nathan Juraj Michlo
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.


import atexit
import settings
import comet_ml
import ray
import ray.tune as tune
import ray.tune.schedulers

from pprint import pprint
from ray.tune.examples.mnist_pytorch_trainable import TrainMNIST as MnistTrainable


# ========================================================================= #
# COMET ML                                                                  #
# ========================================================================= #



EXP = comet_ml.Experiment(
    disabled=(not settings.ENABLE_COMET_ML),
    api_key=settings.COMET_ML_API_KEY,
    project_name=settings.COMET_ML_PROJECT_NAME,
    workspace=settings.COMET_ML_WORKSPACE,
)


# ========================================================================= #
# RAY                                                                       #
# ========================================================================= #


ray.init(address=settings.RAY_ADDRESS)

@atexit.register
def _():
    print('Shutting down')
    ray.shutdown()

# ========================================================================= #
# SCHEDULER                                                                 #
# ========================================================================= #


# scheduler = tune.schedulers.PopulationBasedTraining(
#     metric="mean_accuracy",
#     hyperparam_mutations={
#
#     }
# )

scheduler = tune.schedulers.ASHAScheduler(
    metric="mean_accuracy"
)


# ========================================================================= #
# EXPERIMENT                                                                #
# ========================================================================= #


analysis = tune.run(
    MnistTrainable,
    scheduler=scheduler,
    resources_per_trial=dict(
        cpu=settings.CPUS_PER_NODE,
        gpu=settings.USE_GPU
    ),
    num_samples=settings.POPULATION_SIZE,
    # compares to values returned from train()
    stop=dict(
        mean_accuracy=0.99,
        training_iteration=100,
    ),
    # sampling functions
    config=dict(
        lr=tune.uniform(0.001, 0.1),
        momentum=tune.uniform(0.1, 0.9),
    ),
)


print(f'Best config is: {analysis.get_best_config(metric="mean_accuracy")}')
print(f'All the configs are:')
pprint(analysis.get_all_configs())


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
