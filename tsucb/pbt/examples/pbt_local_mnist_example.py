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



import argparse
import torch
import torch.optim as optim
import numpy as np
import os

from typing import Tuple, NoReturn
from copy import deepcopy
from tsucb.helper.torch.trainable import TorchTrainable
from tsucb.pbt.pbt import Member, Population, IExploiter, Exploiter
from tsucb.pbt.strategies import GeneralisedExploiter, SuggestUniformRandom, ExploitStrategyTruncationSelection

# ========================================================================= #
# MUTATIONS                                                                 #
# ========================================================================= #

def perturb(value, low, high, min, max):
    if np.random.random() < 0.5:
        val = low*value
    else:
        val = high*value
    val = np.clip(val, min, max)
    return val

def uniform_perturb(value, low, high, min, max):
    val = np.random.uniform(low*value, high*value)
    val = np.clip(val, min, max)
    return val

MUTATIONS = {
    'perturb': perturb,
    'uniform_perturb': uniform_perturb,
}

# ========================================================================= #
# MEMBER                                                                    #
# ========================================================================= #

CHECKPOINT_DIR = './checkpoints'
CHECKPOINT_MAP = {}


class MemberTorch(Member):

    def __init__(self, config):
        super().__init__()
        self._trainable = TorchTrainable(config)

    def _save_theta(self, id):
        CHECKPOINT_MAP[id] = os.path.join(CHECKPOINT_DIR, f'checkpoint_{id}.dat')
        self._trainable.save(CHECKPOINT_MAP[id])

    def _load_theta(self, id):
        self._trainable.restore(CHECKPOINT_MAP[id])

    def copy_h(self) -> dict:
        return self._trainable.get_config()
    def _set_h(self, h) -> NoReturn:
        self._trainable = TorchTrainable(config=h)

    def _explored_h(self, population: 'Population') -> dict:
        config = self._trainable.get_config()
        mutations = config['mutations']
        assert len(mutations) > 0
        for path, args in mutations.items():
            current, keys = config, path.split('/')
            for k in keys[:-1]:
                current = current[k]
            current[keys[-1]] = MUTATIONS[args[0]](current[keys[-1]], *args[1:])
        return config

    def _step(self, options: dict) -> NoReturn:
        # TODO this needs to be every N steps, NOT EVERY EPOCH...
        self._trainable.train()
        return None

    def _eval(self, options: dict) -> float:
        results = self._trainable.eval()
        return results['correct']


# ========================================================================= #
# MEMBER                                                                    #
# ========================================================================= #


def main():

    exploiter = GeneralisedExploiter(
        strategy=ExploitStrategyTruncationSelection(),
        suggester=SuggestUniformRandom()
    )

    members = [
        MemberTorch(dict(
            model='example',
            dataset='MNIST',
            loss='MSELoss',

            optimizer='SGD',
            optimizer_options=dict(
                lr=np.random.uniform(0.0001, 0.1),
                momentum=np.random.uniform(0.99, 0.01),
            ),

            mutations={
                'optimizer_options/lr': ('perturb', 0.8, 1.2, 0.0001, 0.1),
                'optimizer_options/momentum': ('perturb', 0.8, 1.2, 0.01, 0.99),
            },

            train_log_interval=-1,
            train_subset=1000,
            batch_size=16,

            use_gpu=True,
        )) for _ in range(10)
    ]

    population = Population(members, exploiter, dict(
        steps_till_ready=1,
        steps=20,
        debug=True,
    ))

    population.train(show_sub_progress=True)


if __name__ == '__main__':
    main()

# ========================================================================= #
# PBT                                                                       #
# ========================================================================= #



















# def main():
    # Training settings
    # parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    # parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
    # parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N', help='input batch size for testing (default: 1000)')
    # parser.add_argument('--epochs', type=int, default=50, metavar='N', help='number of epochs to train (default: 10)')
    # parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='learning rate (default: 0.01)')
    # parser.add_argument('--momentum', type=float, default=0.5, metavar='M', help='SGD momentum (default: 0.5)')
    # parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    # parser.add_argument('--seed', type=int, default=42, metavar='S', help='random seed (default: 1)')
    # parser.add_argument('--log-interval', type=int, default=468, metavar='N', help='how many batches to wait before logging training status')
    # parser.add_argument('--save-model', action='store_true', default=False, help='For Saving the current Model')
    # args = parser.parse_args()
    #
    # use_cuda = True # not args.no_cuda and torch.cuda.is_available()
    # torch.manual_seed(args.seed)
    # device = torch.device("cuda" if use_cuda else "cpu")


    # config = {
    #     'use_gpu': True,
    #     # dataset
    #     'num_workers': 1,
    #     'pin_memory': True,
    #     # makers
    #     'make_dataset': get_dataset_loaders_maker('MNIST'),
    #     'make_model': get_model_maker('MNIST'),
    # }

    # model = Net().to(device)
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    #
    # for epoch in range(1, args.epochs + 1):
    #     train(args, model, device, train_loader, optimizer, epoch)
    #     test(args, model, device, test_loader)
    #
    # if (args.save_model):
    #     torch.save(model.state_dict(), "mnist_cnn.pt")







# from copy import deepcopy
# from typing import NamedTuple, NoReturn
# import numpy as np
# import matplotlib.pyplot as plt
# import ray
# from ray.tune.examples.mnist_pytorch import ConvNet, get_data_loaders, train, test
# from tqdm import tqdm
# from tsucb.pbt.strategies import ExploitUcb, OrigExploitTruncationSelection
# from tsucb.pbt.pbt import Member, Population
# import os
# import ray.tune as tune
#
# import torch
#
#
# # ========================================================================= #
# # SYSTEM                                                                    #
# # ========================================================================= #
#
#
# class ToyHyperParams(NamedTuple):
#     learning_rate: float
#
#
# class TrainMNIST(tune.Trainable):
#     def _setup(self, config):
#         use_cuda = config.get("use_gpu") and torch.cuda.is_available()
#         self.device = torch.device("cuda" if use_cuda else "cpu")
#         self.train_loader, self.test_loader = get_data_loaders()
#         self.model = ConvNet().to(self.device)
#         self.optimizer = torch.optim.SGD(
#             self.model.parameters(),
#             lr=config.get("lr", 0.01),
#             momentum=config.get("momentum", 0.9))
#
#     def eval(self):
#         return test(self.model, self.test_loader, self.device)
#
#     def _train(self):
#         train(self.model, self.optimizer, self.train_loader, device=self.device)
#         return {"mean_accuracy": self.eval()}
#
#     def _save(self, checkpoint_dir):
#         checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
#         torch.save(self.model.state_dict(), checkpoint_path)
#         return checkpoint_path
#
#     def _restore(self, checkpoint_path):
#         self.model.load_state_dict(torch.load(checkpoint_path))
#
#
# # ========================================================================= #
# # Member - TOY                                                              #
# # ========================================================================= #
#
# CHECKPOINT_MAP = {}
#
#
# class MemberMnist(Member):
#
#     def __init__(self):
#         super().__init__()
#         config = dict(
#             lr=np.random.uniform(0.001, 0.1),
#             momentum=np.random.uniform(0.1, 0.9),
#             use_gpu=True,
#         )
#         self._trainable = TrainMNIST(config=config)
#
#     def _save_theta(self, id):
#         CHECKPOINT_MAP[id] = self._trainable.save(f'./checkpoints/{id}')
#     def _load_theta(self, id):
#         self._trainable.restore(CHECKPOINT_MAP[id])
#
#     def copy_h(self) -> dict:
#         return deepcopy(self._trainable.config)
#     def _set_h(self, h) -> NoReturn:
#         self._trainable = TrainMNIST(config=h)
#         # self._trainable.config = h  # I think this is all you need, but am not sure.
#     def _explored_h(self, population: 'Population') -> dict:
#         return dict(
#            lr=np.random.uniform(0.001, 0.1),
#            momentum=np.random.uniform(0.1, 0.9),
#         )
#
#     def _step(self, options: dict) -> np.ndarray:
#         result = self._trainable.train()
#         return result
#     def _eval(self, options: dict) -> float:
#         return self._trainable.eval()
#
# # ============================================================================ #
# # PLOTTING                                                                     #
# # ============================================================================ #
#
#
# #
# # def make_subplots(h, w, figsize=None):
# #     fig, axs = plt.subplots(h, w, figsize=figsize)
# #     return fig, np.array(axs).reshape((h, w))
# #
# # def plot_performance(ax, population, steps, title):
# #     for member, color in zip(population, 'brgcmyk'):
# #         vals = [step.p for step in member]
# #         ax.plot(vals, color=color, lw=0.7)
# #     ax.axhline(y=1.2, linestyle='dotted', color='k')
# #     ax.set(xlim=[0, steps-1], ylim=[-0.5, 1.31], title=title, xlabel='Step', ylabel='Q')
# #
# # def plot_theta(ax, population, steps, title):
# #     for member, color in zip(population, 'brgcmyk'):
# #         x, y = np.array([step.theta[0] for step in member]), np.array([step.theta[1] for step in member])
# #         jumps = np.where([step.exploit_id is not None for step in member])[0]
# #         x, y = np.insert(x, jumps, np.nan), np.insert(y, jumps, np.nan)
# #         ax.plot(x, y, color=color, lw=0.5, zorder=1)
# #         ax.scatter(x, y, color=color, s=1, zorder=2)
# #
# #     ax.set(xlim=[-0.1, 1], ylim=[-0.1, 1], title=title, xlabel='theta0', ylabel='theta1')
#
# def experiment(options, exploiter, n=20, steps=200, exploit=True, explore=True, title=None):
#     population = Population([MemberMnist() for i in range(n)], exploiter=exploiter, options=options)
#     population.train(steps, exploit=exploit, explore=explore)
#
#     # Calculates the score as the index of the first occurrence greater than 1.18
#     # scores = np.array([[h.p for h in m] for m in population])
#     # firsts = np.argmax(scores > 1.18, axis=1)
#     # firsts[firsts == 0] = scores.shape[1]
#     # time_to_converge = np.min(firsts)
#
#     # score = np.max(population.scores)
#
#     # plot_theta(ax_col[0], population, steps=steps, title=title)
#     # plot_performance(ax_col[1], population, steps=steps, title=title)
#     return np.max(population.scores), np.average(np.array([[h.p for h in m] for m in population]), axis=0)
#
#
# if __name__ == '__main__':
#
#     ray.init()
#
#     options = {
#         "steps": 20,
#         "steps_till_ready": 2,
#         "exploration_scale": 0.1,
#     }
#
#     # REPEAT EXPERIMENT N TIMES
#     n, k, repeats = 10, 2, 1
#     score, scores = np.zeros(k), np.zeros((k, options['steps']))
#     # fig, axs = make_subplots(2, len(scores))
#
#     with tqdm(range(repeats)) as itr:
#         for i in itr:
#             score_0, scores_0 = experiment(options, OrigExploitTruncationSelection(), n=n, steps=options["steps"], exploit=True, explore=True, title='PBT Trunc Sel')
#             score_1, scores_1 = experiment(options, ExploitUcb(subset_mode='top', incr_mode='exploited', normalise_mode='subset'),  n=n, steps=options["steps"], exploit=True, explore=True, title='PBT Ucb Sel')
#
#             score += [score_0, score_1]
#             scores += [scores_0, scores_1]
#
#             print(score_0)
#             print(scores_0)
#             print(score_1)
#             print(scores_1)
#
#             # itr.set_description(f'{np.around(scores / (i + 1), 4)} | {np.around(converge_times / (i + 1), 2)}')
#     scores /= repeats
#     score /= repeats
#
#     print(f'T: {score[0]} | {list(scores[0])}')
#     print(f'U: {score[1]} | {list(scores[1])}')
#
#     fig, ax = plt.subplots(1, 1)
#
#     ax.plot(scores[0], label='PBT Trunc Sel')
#     ax.plot(scores[1], label='PBT Ucb Sel')
#     ax.legend()
#     ax.set(title=f'Trunc vs Ucb: {dict(n=n, r=options["steps_till_ready"])}', xlabel='Steps', ylabel='Ave Max Score')
#
#     fig.show()
#
#





# class StepTrainer(object):
#     def __init__(self, train_loader):
#         self._train_loader = train_loader
#         self._batch_iter = None
#
#     def train(self, model, device, train_loader, optimizer, num_examples=None):
#         model.train()
#
#         if num_examples is None:
#             num_examples = len(train_loader.dataset)
#
#         for i in range(0, num_examples, self._train_loader.batch_size):
#             try:
#                 batch_i, (data, target) = next(self._batch_iter)
#             except:
#                 print('[RESET ITER]')
#                 self._batch_iter = enumerate(train_loader)
#                 batch_i, (data, target) = next(self._batch_iter)
#
#             data, target = data.to(device), target.to(device)
#             optimizer.zero_grad()
#             output = model(data)
#             loss = F.nll_loss(output, target)
#             loss.backward()
#             optimizer.step()
#
#             self.log(i=(batch_i+1)*self._train_loader.batch_size, n=len(train_loader.dataset), loss=loss)
#
#     @util.min_time_elapsed(0.5)
#     def log(self, i, n, loss):
#         print(
#             f'[{i}/{n} {100.*i/n:.1f}%] Loss: {loss.item():.6f}')
