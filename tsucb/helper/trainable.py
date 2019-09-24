
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


# import ray
# from ray.experimental.sgd.tf import TFTrainable, TFTrainer
# import tensorflow_datasets as tfds
# from tsucb.helper import models
#
#
# # ========================================================================= #
# # TEMP                                                                     #
# # ========================================================================= #
#
# NUM_TRAIN_SAMPLES = 1000
# NUM_TEST_SAMPLES = 400
#
# def simple_dataset(batch_size=16):
#     from tensorflow.data import Dataset
#
#     def linear_dataset(size):
#         import numpy as np
#         x = np.random.rand(size, 28 * 28 * 1)
#         y = np.random.rand(size, 10)
#         return x, y
#
#     x_train, y_train = linear_dataset(size=NUM_TRAIN_SAMPLES)
#     x_test, y_test = linear_dataset(size=NUM_TEST_SAMPLES)
#
#     train_dataset = Dataset.from_tensor_slices((x_train, y_train))
#     test_dataset = Dataset.from_tensor_slices((x_test, y_test))
#     train_dataset = train_dataset.shuffle(NUM_TRAIN_SAMPLES).repeat().batch(batch_size)
#     test_dataset = test_dataset.repeat().batch(batch_size)
#
#     return train_dataset, test_dataset
#
#
# # ========================================================================= #
# # trainable                                                                 #
# # ========================================================================= #
#
#
#
# def _model_creator(config):
#     # if 'example' == config['model']:
#     #     if config['dataset'] in {'mnist', 'random'}:
#     model = models.create_mnist_model('channels_first' if config['use_gpu'] else 'channels_last')
#     #     else:
#     #         raise KeyError('Only dataset="mnist" is currently supported for model="example"')
#     # else:
#     #     raise KeyError('Only model="example" is currently available.')
#
#     model.compile(
#         optimizer=config['optimizer'],
#         loss=config['loss'],
#         metrics=config.get('metrics', [config['loss']])
#     )
#
#     return model
#
#
# def _data_creator(config, return_info=False):
#     """
#     Examples: iris, cifar10, cifar100, mnist, emnist, kmnist, fashion_mnist
#     """
#     # if config['dataset'] == 'random':
#     return simple_dataset(config["batch_size"])
#
#     # data, info = tfds.load(config['dataset'], with_info=True)
#     # if return_info:
#     #     return info
#     # else:
#     #     return data['train'], data['test']
#
#
# class GeneralTrainable(TFTrainable):
#
#     # def __init__(self, config: dict = None, logger_creator=None):
#     #     # dataset_info = _data_creator(config, return_info=True)
#     #     # print(f'\n[DATASET]:\n{dataset_info}\n')
#     #     # SET BATCH SIZE DEFAULTS
#     #     # fit_config = config.setdefault('fit_config', {})
#     #     # evaluate_config = config.setdefault('evaluate_config', {})
#     #     # GET LENGTH
#     #     # fit_config.setdefault('steps_per_epoch', dataset_info.splits['train'].num_examples // config['batch_size'])
#     #     # fit_config.setdefault('steps_per_epoch', 1000)  # TODO: fix... this is intended to help with infinitely repeating datasets
#     #     # evaluate_config.setdefault('steps', dataset_info.splits['test'].num_examples // config['batch_size'])
#     #     # evaluate_config.setdefault('steps', 1000)  # TODO: fix... this is intended to help with infinitely repeating datasets
#     #     # INITIALISE PARENT
#     #     super().__init__(config=config, logger_creator=logger_creator)
#
#     def _setup(self, config):
#         self._trainer = TFTrainer(
#             model_creator=_model_creator,
#             data_creator=_data_creator,
#             config=config,
#             num_replicas=config['num_replicas'],
#             use_gpu=config['use_gpu']
#         )
#
#
# # ========================================================================= #
# # END                                                                       #
# # ========================================================================= #
#
#
# ray.init()
#
#
# trainable = GeneralTrainable(dict(
#     # trainer config
#     # trainer_config=dict(
#         model='example',
#         optimizer='sgd',
#         loss='mse',
#         dataset='random',
#
#         batch_size=16,
#         fit_config={
#             "steps_per_epoch": NUM_TRAIN_SAMPLES // 16
#         },
#         evaluate_config={
#             "steps": NUM_TEST_SAMPLES // 16,
#         },
#     # ),
#     # trainable config
#     use_gpu=True,
#     num_replicas=1,
# ))
#
# trainable.train()




import tensorflow_datasets as tfds
import argparse

from ray.experimental.sgd.tf.tf_runner import TFRunner
from tensorflow.data import Dataset
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

import ray
from ray import tune
from ray.experimental.sgd.tf.tf_trainer import TFTrainer, TFTrainable
from tensorflow.python.data.ops.dataset_ops import _OptionsDataset, DatasetV1Adapter
from tensorflow.python.keras.datasets.mnist import load_data
from tensorflow.python.ops.gen_dataset_ops import BatchDataset
import tensorflow as tf

from helper import models

NUM_TRAIN_SAMPLES = 1000
NUM_TEST_SAMPLES = 400


def create_config(batch_size):
    return {
        "batch_size": batch_size,
        "fit_config": {
            "steps_per_epoch": NUM_TRAIN_SAMPLES // batch_size
        },
        "evaluate_config": {
            "steps": NUM_TEST_SAMPLES // batch_size,
        }
    }




def simple_dataset(config):
    def linear_dataset(size=1000):
        x = np.random.rand(size, 28 * 28 * 1)
        y = np.random.rand(size, 10)
        return x, y

    batch_size = config["batch_size"]
    x_train, y_train = linear_dataset(size=NUM_TRAIN_SAMPLES)
    x_test, y_test = linear_dataset(size=NUM_TEST_SAMPLES)

    train_dataset = Dataset.from_tensor_slices((x_train, y_train))
    test_dataset = Dataset.from_tensor_slices((x_test, y_test))
    train_dataset = train_dataset.shuffle(NUM_TRAIN_SAMPLES).repeat().batch(batch_size)
    test_dataset = test_dataset.repeat().batch(batch_size)

    for i in test_dataset:
        print(i)
        break

    train_ds = tfds.load("mnist", split=tfds.Split.TRAIN, batch_size=-1)
    train_np = tfds.as_numpy(train_ds)
    x_test, y_test = train_np["image"].reshape((-1, 28*28)), train_np["label"].reshape((-1, 10))


    test_ds = tfds.load("mnist", split=tfds.Split.TEST, batch_size=-1)
    test_np = tfds.as_numpy(train_ds)
    x_test, y_test = test_np["image"].reshape((-1, 28*28)), test_np["label"].reshape((-1, 1))

    print(test_np["image"].shape)
    print(x_test.shape)
    print(test_np["label"].shape)
    print(y_test.shape)


    train_dataset = Dataset.from_tensor_slices((x_train, y_train))
    test_dataset = Dataset.from_tensor_slices((x_test, y_test))
    train_dataset = train_dataset.shuffle(NUM_TRAIN_SAMPLES).repeat().batch(batch_size)
    test_dataset = test_dataset.repeat().batch(batch_size)

    return train_dataset, test_dataset

def simple_model(config):
    model = models.create_mnist_model('channels_last')

    model.compile(
        optimizer="sgd",
        loss="mean_squared_error",
        metrics=["mean_squared_error"]
    )

    return model


def train_example(num_replicas=1, batch_size=128, use_gpu=False):
    trainer = TFTrainer(
        model_creator=simple_model,
        data_creator=simple_dataset,
        num_replicas=num_replicas,
        use_gpu=use_gpu,
        verbose=True,
        config=create_config(batch_size)
    )

    train_stats1 = trainer.train()
    train_stats1.update(trainer.validate())
    print(train_stats1)

    train_stats2 = trainer.train()
    train_stats2.update(trainer.validate())
    print(train_stats2)

    val_stats = trainer.validate()
    print(val_stats)
    print("success!")


def tune_example(num_replicas=1, use_gpu=False):
    config = {
        "model_creator": simple_model,
        "data_creator": simple_dataset,
        "num_replicas": num_replicas,
        "use_gpu": use_gpu,
        "trainer_config": create_config(batch_size=128)
    }

    analysis = TFTrainable(config).train()

    # analysis = tune.run(
    #     TFTrainable,
    #     num_samples=2,
    #     config=config,
    #     stop={"training_iteration": 2},
    #     verbose=1
    # )

    print(analysis)
    # print(analysis.get_best_config(metric="validation_loss", mode="min"))

    # return analysis.get_best_config(metric="validation_loss", mode="min")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--redis-address", required=False, type=str, help="the address to use for Redis")
    parser.add_argument("--num-replicas", "-n", type=int, default=1, help="Sets number of replicas for training.")
    parser.add_argument("--use-gpu", action="store_true", default=False, help="Enables GPU training")
    parser.add_argument("--tune", action="store_true", default=False, help="Tune training")

    args, _ = parser.parse_known_args()

    ray.init(redis_address=args.redis_address)

    # if args.tune:
    tune_example(num_replicas=args.num_replicas, use_gpu=args.use_gpu)
    # else:
    #     train_example(num_replicas=args.num_replicas, use_gpu=args.use_gpu)



