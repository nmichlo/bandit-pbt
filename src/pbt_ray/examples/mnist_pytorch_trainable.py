#!/usr/bin/env python

import argparse
from pprint import pprint
from random import random
import ray
from ray import tune
from ray.tune.examples.mnist_pytorch_trainable import TrainMNIST
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining


# Change these values if you want the training to run quicker or slower.
EPOCH_SIZE = 512
TEST_SIZE = 256

# Training settings
parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
parser.add_argument("--use-gpu", action="store_true", default=True, help="enables CUDA training")
parser.add_argument("--ray-redis-address", type=str, help="The Redis address of the cluster.")
parser.add_argument("--smoke-test", action="store_true", help="Finish quickly for testing")
parser.add_argument("--sched", type=str, default='pbt', help="Specify the scheduler {pbt, asha}")


if __name__ == "__main__":
    args = parser.parse_args()
    ray.init(redis_address=args.ray_redis_address)

    if args.sched == 'pbt':
        sched = PopulationBasedTraining(
            time_attr="training_iteration",
            metric="mean_accuracy",
            mode="max",
            perturbation_interval=20,
            hyperparam_mutations={
                "lr": lambda: random.uniform(0.0001, 0.02),
                "momentum": lambda: random.uniform(0.99, 0.01),
            }
        )
    elif args.sched == 'asha':
        sched = ASHAScheduler(metric="mean_accuracy")
    else:
        raise KeyError('Invalid scheduler!')

    analysis = tune.run(
        TrainMNIST,
        scheduler=sched,
        # args
        stop=dict(
            mean_accuracy=0.99,
            training_iteration=5,
        ),
        resources_per_trial=dict(
            cpu=3,
            gpu=int(args.use_gpu)
        ),
        num_samples=2,
        checkpoint_at_end=True,
        checkpoint_freq=3,
        config=dict(
            args=args,
            lr=tune.uniform(0.001, 0.1),
            momentum=tune.uniform(0.1, 0.9),
        )
    )

    print("Best config is:", analysis.get_best_config(metric="mean_accuracy"))
    print('All configs:')
    pprint(analysis.get_all_configs())