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


import torchvision
import torch.nn as nn
import torch.nn.modules
import torch.nn.functional as F

from tsucb.helper import util


# ========================================================================= #
# HELPER                                                                    #
# ========================================================================= #


def train(model, device, train_loader, optimizer, log_time_interval=1):
    """
    FROM: https://github.com/pytorch/examples/blob/master/mnist/main.py
    """
    if log_time_interval > 0:
        @util.min_time_elapsed(log_time_interval)
        def _log(batch_i, loader, loss):
            print(f'[{batch_i*loader.batch_size}/{len(loader.dataset)} {100.*batch_i/len(loader):.1f}%] Loss: {loss.item():.6f}')

    model.train()
    for batch_i, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        # log
        if log_time_interval > 0:
            _log(batch_i, train_loader, loss)


def test(model, device, test_loader):
    """
    FROM: https://github.com/pytorch/examples/blob/master/mnist/main.py
    """
    model.eval()
    correct, test_loss = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    return correct / len(test_loader.dataset), test_loss


# ========================================================================= #
# example                                                                   #
# ========================================================================= #


class MnistModel(nn.Module):
    """
    FROM: https://github.com/pytorch/examples/blob/master/mnist/main.py
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5, stride=1)   # size: (28 - (5-1)) / 2 = 12
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=50, kernel_size=5, stride=1)  # size: (12 - (5-1)) / 2 = 4
        self.fc1 = nn.Linear(in_features=4*4*50, out_features=500)
        self.fc2 = nn.Linear(in_features=500, out_features=10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# ========================================================================= #
# creator                                                                   #
# ========================================================================= #


def create_torchvision_model(arch, num_classes, **kwargs):
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
# MODEL MAKER                                                                #
# ========================================================================= #


# ========================================================================= #
# END                                                                       #
# ========================================================================= #


