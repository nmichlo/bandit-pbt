
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


from pprint import pprint
import torchvision
import torch.nn as nn
import torch.nn.modules
import torch.nn.functional as F
from torch.optim.sparse_adam import SparseAdam

import tensorflow as tf
from helper import util


# ========================================================================= #
# example                                                                   #
# ========================================================================= #


class ConvNetExample(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        assert num_classes == 10

        self.conv1 = nn.Conv2d( 1, 32, 5, 1, padding=2)
        self.conv2 = nn.Conv2d(32, 32, 5, 1, padding=2)
        self.fc1 = nn.Linear(7 * 7 * 32, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        print(x.shape)
        x = F.relu(self.conv1(x))
        print(x.shape)
        x = F.max_pool2d(x, 2, 2)
        print(x.shape)
        x = F.relu(self.conv2(x))
        print(x.shape)
        x = F.max_pool2d(x, 2, 2)
        print(x.shape)
        x = x.view(-1, 7 * 7 * 32)
        print(x.shape)
        x = F.relu(self.fc1(x))
        print(x.shape)
        x = self.fc2(x)
        print(x.shape)
        return F.log_softmax(x, dim=1)



# ========================================================================= #
# creator                                                                   #
# ========================================================================= #


# def create_model(arch, num_classes, **kwargs):
#     """
#     torchvision.modules:
#         - Valid architectures include: [Signature (pretrained=False, progress=True, **kwargs)]
#             ['alexnet', 'densenet121', 'densenet161', 'densenet169', 'densenet201', 'googlenet',
#              'inception_v3', 'mnasnet0_5', 'mnasnet0_75', 'mnasnet1_0', 'mnasnet1_3', 'mobilenet_v2',
#              'resnet101', 'resnet152', 'resnet18', 'resnet34', 'resnet50', 'resnext101_32x8d',
#              'resnext50_32x4d', 'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5',
#              'shufflenet_v2_x2_0', 'squeezenet1_0', 'squeezenet1_1', 'vgg11', 'vgg11_bn', 'vgg13',
#              'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn', 'wide_resnet101_2', 'wide_resnet50_2'
#          - Valid Models Include: [Signature (num_classes=1000, ...)]
#             ['AlexNet', 'DenseNet', 'GoogLeNet', 'Inception3', 'MNASNet',
#              'MobileNetV2', 'ResNet', 'ShuffleNetV2', 'SqueezeNet', 'VGG']
#          - Valid submodules include:
#             MODELS: ['densenet', 'inception', 'mnasnet', 'mobilenet', 'resnet', 'shufflenetv2', 'squeezenet', 'vgg']
#             OTHER: [ 'detection', 'segmentation', 'utils', 'video']
#
#     :param arch: One of the architectures listed above.
#     :param num_classes: The number of target classes.
#     :param kwargs: Additional arguments passed to the Model.
#     :return: torch.nn.Model
#     """
#     if arch == 'example':
#         assert not kwargs, 'kwargs are not supported for the example'
#         return ConvNetExample(num_classes=num_classes)
#
#     model_builders = util.get_module_functions(torchvision.models)
#     assert arch in model_builders, f'arch "{arch}" not in {sorted(model_builders.keys())}'
#
#     model_builder = model_builders[arch]
#     return model_builder(pretrained=False, progress=True, num_classes=num_classes, **kwargs)

# ========================================================================= #
# END                                                                       #
# ========================================================================= #


