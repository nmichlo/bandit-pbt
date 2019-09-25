
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

#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:


import torchvision
import torch.nn as nn
import torch.nn.modules

from helper import util


# ========================================================================= #
# example                                                                   #
# ========================================================================= #


def create_mnist_model():
    """
    From https://github.com/tensorflow/models/blob/master/official/mnist/mnist.py
    :param data_format: 'channels_first' is typically faster on GPUs while 'channels_last' is typically faster on CPUs.
    :return: A torch.nn.Sequential model
    """
    return torch.nn.Sequential(
        nn.Conv2d(28*28*1, 32, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        nn.Conv2d(32, 64, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        nn.Linear(7*7*64, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.4),
        nn.Linear(1024, 10),
            nn.LogSoftmax(dim=1),
    )


# ========================================================================= #
# creator                                                                   #
# ========================================================================= #


def create_model(arch, num_classes, **kwargs):
    """
    torchvision.modules:
        - Valid architectures include: [Signature (pretrained=False, progress=True, **kwargs)]
            ['alexnet', 'densenet121', 'densenet161', 'densenet169', 'densenet201', 'googlenet',
             'inception_v3', 'mnasnet0_5', 'mnasnet0_75', 'mnasnet1_0', 'mnasnet1_3', 'mobilenet_v2',
             'resnet101', 'resnet152', 'resnet18', 'resnet34', 'resnet50', 'resnext101_32x8d',
             'resnext50_32x4d', 'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5',
             'shufflenet_v2_x2_0', 'squeezenet1_0', 'squeezenet1_1', 'vgg11', 'vgg11_bn', 'vgg13',
             'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn', 'wide_resnet101_2', 'wide_resnet50_2'
         - Valid Models Include: [Signature (num_classes=1000, ...)]
            ['AlexNet', 'DenseNet', 'GoogLeNet', 'Inception3', 'MNASNet',
             'MobileNetV2', 'ResNet', 'ShuffleNetV2', 'SqueezeNet', 'VGG']
         - Valid submodules include:
            MODELS: ['densenet', 'inception', 'mnasnet', 'mobilenet', 'resnet', 'shufflenetv2', 'squeezenet', 'vgg']
            OTHER: [ 'detection', 'segmentation', 'utils', 'video']

    :param arch: One of the architectures listed above.
    :param num_classes: The number of target classes.
    :param kwargs: Additional arguments passed to the Model.
    :return: torch.nn.Model
    """
    model_builders = util.get_module_functions(torchvision.models)
    assert arch in model_builders, f'arch "{arch}" not in {sorted(model_builders.keys())}'

    model_builder = model_builders[arch]
    return model_builder(pretrained=False, progress=True, num_classes=num_classes, **kwargs)


# ========================================================================= #
# END                                                                       #
# ========================================================================= #


