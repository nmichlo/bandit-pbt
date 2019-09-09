#!/usr/bin/env python
# https://github.com/ray-project/ray/blob/master/python/ray/tune/examples/mnist_pytorch_trainable.py

# Original Code here:
# https://github.com/pytorch/examples/blob/master/mnist/main.py

from __future__ import print_function

import argparse
import os
from random import random

import torch
import torch.optim as optim

import ray
from ray import tune
from ray.tune import ExperimentAnalysis
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining

# Change these values if you want the training to run quicker or slower.
from examples.mnist_pytorch import (train, test, get_data_loaders, ConvNet)

EPOCH_SIZE = 512
TEST_SIZE = 256

# Training settings
parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
parser.add_argument(
    "--use-gpu",
    action="store_true",
    default=True,
    help="enables CUDA training")
parser.add_argument(
    "--ray-redis-address", type=str, help="The Redis address of the cluster.")
parser.add_argument(
    "--smoke-test", action="store_true", help="Finish quickly for testing")


# Below comments are for documentation purposes only.
# yapf: disable
# __trainable_example_begin__
class TrainMNIST(tune.Trainable):
    def _setup(self, config):
        use_cuda = config.get("use_gpu") and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.train_loader, self.test_loader = get_data_loaders()
        self.model = ConvNet().to(self.device)
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=config.get("lr", 0.01),
            momentum=config.get("momentum", 0.9)
        )

    def _train(self):
        train(
            self.model, self.optimizer, self.train_loader, device=self.device)
        acc = test(self.model, self.test_loader, self.device)
        return {"mean_accuracy": acc}

    def _save(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
        torch.save(self.model.state_dict(), checkpoint_path)
        return checkpoint_path

    def _restore(self, checkpoint_path):
        self.model.load_state_dict(torch.load(checkpoint_path))


# __trainable_example_end__
# yapf: enable

if __name__ == "__main__":
    args = parser.parse_args()
    ray.init(redis_address=args.ray_redis_address)


    sched = ASHAScheduler(metric="mean_accuracy")

    # sched = PopulationBasedTraining(
    #     time_attr="training_iteration",
    #     metric="mean_accuracy",
    #     mode="max",
    #     perturbation_interval=20,
    #     hyperparam_mutations={
    #         "lr": lambda: random.uniform(0.0001, 0.02),
    #         "momentum": lambda: random.uniform(0.99, 0.01),
    #     }
    # )

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

    dfs = analysis.get_all_configs()
    print(dfs)
    # This plots everything on the same plot
    # ax = None
    # for d in dfs.values():
    #     ax = d.plot("timestamp", "mean_accuracy", ax=ax, legend=False)

    # while True:
    # try:
        # print('A', analysis.trial_dataframes)
    # for obj in analysis.fetch_trial_dataframes():
    #     print('OBJ1', obj)
    #
    # for obj in analysis.trails:
    #     print('OBJ2', obj)

        # ax = None
        # for df in analysis.trial_dataframes:
            # ax = df.mean_accuracy.plot(ax=ax, legend=False)
    # except:
    #     pass

    print("Best config is:", analysis.get_best_config(metric="mean_accuracy"))