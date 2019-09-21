
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

# ========================================================================= #
# info                                                                   #
# ========================================================================= #


def print_torchvision_model_heirarchy():
    """
    Module
      - AlexNet              (num_classes=1000)
      - DenseNet             (growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000, memory_efficient=False)
      - GoogLeNet            (num_classes=1000, aux_logits=True, transform_input=False, init_weights=True)
      - Inception3           (num_classes=1000, aux_logits=True, transform_input=False)
      - MNASNet              (alpha, num_classes=1000, dropout=0.2)
      - MobileNetV2          (num_classes=1000, width_mult=1.0, inverted_residual_setting=None, round_nearest=8)
      - ResNet               (block, layers, num_classes=1000, zero_init_residual=False, groups=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=None)
      - ShuffleNetV2         (stages_repeats, stages_out_channels, num_classes=1000)
      - SqueezeNet           (version='1_0', num_classes=1000)
      - VGG                  (features, num_classes=1000, init_weights=True)
    """

    from helper import util
    import torchvision.models
    util.print_module_class_heirarchy(torchvision.models, 'Module')


def print_torch_nn_loss_heirarchy():
    """
    _Loss                (size_average=None, reduce=None, reduction='mean')
      - BCEWithLogitsLoss    (weight=None, size_average=None, reduce=None, reduction='mean', pos_weight=None)
      - CTCLoss              (blank=0, reduction='mean', zero_infinity=False)
      - CosineEmbeddingLoss  (margin=0.0, size_average=None, reduce=None, reduction='mean')
      - HingeEmbeddingLoss   (margin=1.0, size_average=None, reduce=None, reduction='mean')
      - KLDivLoss            (size_average=None, reduce=None, reduction='mean')
      - L1Loss               (size_average=None, reduce=None, reduction='mean')
      - MSELoss              (size_average=None, reduce=None, reduction='mean')
      - MarginRankingLoss    (margin=0.0, size_average=None, reduce=None, reduction='mean')
      - MultiLabelMarginLoss (size_average=None, reduce=None, reduction='mean')
      - PoissonNLLLoss       (log_input=True, full=False, size_average=None, eps=1e-08, reduce=None, reduction='mean')
      - SmoothL1Loss         (size_average=None, reduce=None, reduction='mean')
      - SoftMarginLoss       (size_average=None, reduce=None, reduction='mean')
      - TripletMarginLoss    (margin=1.0, p=2.0, eps=1e-06, swap=False, size_average=None, reduce=None, reduction='mean')
      - _WeightedLoss        (weight=None, size_average=None, reduce=None, reduction='mean')
          - BCELoss              (weight=None, size_average=None, reduce=None, reduction='mean')
          - CrossEntropyLoss     (weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean')
          - MultiLabelSoftMarginLoss (weight=None, size_average=None, reduce=None, reduction='mean')
          - MultiMarginLoss      (p=1, margin=1.0, weight=None, size_average=None, reduce=None, reduction='mean')
          - NLLLoss              (weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean')
              - NLLLoss2d            (weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean')
    """
    from helper import util
    import torch.nn.modules.loss
    util.print_module_class_heirarchy(torch.nn.modules.loss, '_Loss')


def print_torch_optim_heirarchy():
    """
    Optimizer            (params, defaults)
      - ASGD                 (params, lr=0.01, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0)
      - Adadelta             (params, lr=1.0, rho=0.9, eps=1e-06, weight_decay=0)
      - Adagrad              (params, lr=0.01, lr_decay=0, weight_decay=0, initial_accumulator_value=0)
      - Adam                 (params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
      - AdamW                (params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
      - Adamax               (params, lr=0.002, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
      - LBFGS                (params, lr=1, max_iter=20, max_eval=None, tolerance_grad=1e-05, tolerance_change=1e-09, history_size=100, line_search_fn=None)
      - RMSprop              (params, lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
      - Rprop                (params, lr=0.01, etas=(0.5, 1.2), step_sizes=(1e-06, 50))
      - SGD                  (params, lr=<required parameter>, momentum=0, dampening=0, weight_decay=0, nesterov=False)
      - SparseAdam           (params, lr=0.001, betas=(0.9, 0.999), eps=1e-08)
    """
    from helper import util
    import torch.optim
    util.print_module_class_heirarchy(torch.optim, 'Optimizer')


def print_torchvision_datasets_heirarchy():
    """
    * = has 'classes' field
    + = train, download, transform

    VisionDataset        (root, transforms=None, transform=None, target_transform=None)
      + CIFAR10              (root, train=True, transform=None, target_transform=None, download=False)
          + CIFAR100             (root, train=True, transform=None, target_transform=None, download=False)
      - Caltech101           (root, target_type='category', transform=None, target_transform=None, download=False)
      - Caltech256           (root, transform=None, target_transform=None, download=False)
      - CelebA               (root, split='train', target_type='attr', transform=None, target_transform=None, download=False)
      - *Cityscapes          (root, split='train', mode='fine', target_type='instance', transform=None, target_transform=None, transforms=None)
      - CocoCaptions         (root, annFile, transform=None, target_transform=None, transforms=None)
      - CocoDetection        (root, annFile, transform=None, target_transform=None, transforms=None)
      - DatasetFolder        (root, loader, extensions=None, transform=None, target_transform=None, is_valid_file=None)
          - ImageFolder          (root, transform=None, target_transform=None, loader=<function default_loader at 0x7f4e4355cc80>, is_valid_file=None)
              - ImageNet             (root, split='train', download=False, **kwargs)
      - FakeData             (size=1000, image_size=(3, 224, 224), num_classes=10, transform=None, target_transform=None, random_offset=0)
      - Flickr30k            (root, ann_file, transform=None, target_transform=None)
      - Flickr8k             (root, ann_file, transform=None, target_transform=None)
      - HMDB51               (root, annotation_path, frames_per_clip, step_between_clips=1, fold=1, train=True, transform=None)
      - Kinetics400          (root, frames_per_clip, step_between_clips=1, transform=None)
      - LSUN                 (root, classes='train', transform=None, target_transform=None)
      - LSUNClass            (root, transform=None, target_transform=None)
      + *MNIST               (root, train=True, transform=None, target_transform=None, download=False)
          + *EMNIST              (root, split, **kwargs)
          + *FashionMNIST        (root, train=True, transform=None, target_transform=None, download=False)
          + *KMNIST              (root, train=True, transform=None, target_transform=None, download=False)
          + *QMNIST              (root, what=None, compat=True, train=True, **kwargs)
      - Omniglot             (root, background=True, transform=None, target_transform=None, download=False)
      + PhotoTour            (root, name, train=True, transform=None, download=False)
      - SBDataset            (root, image_set='train', mode='boundaries', download=False, transforms=None)
      - SBU                  (root, transform=None, target_transform=None, download=True)
      - SEMEION              (root, transform=None, target_transform=None, download=True)
      - STL10                (root, split='train', folds=None, transform=None, target_transform=None, download=False)
      - SVHN                 (root, split='train', transform=None, target_transform=None, download=False)
      - UCF101               (root, annotation_path, frames_per_clip, step_between_clips=1, fold=1, train=True, transform=None)
      + USPS                 (root, train=True, transform=None, target_transform=None, download=False)
      - VOCDetection         (root, year='2012', image_set='train', download=False, transform=None, target_transform=None, transforms=None)
      - VOCSegmentation      (root, year='2012', image_set='train', download=False, transform=None, target_transform=None, transforms=None)
    """
    from helper import util
    import torchvision
    util.print_module_class_heirarchy(torchvision.datasets, 'VisionDataset')


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
