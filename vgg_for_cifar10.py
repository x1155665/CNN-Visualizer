'''
vgg models modified for cifar 10 (32x32)
fc-dimension: 512, 512, 10
The layers are named.

Modified from https://github.com/pytorch/vision.git
Modified from https://github.com/chengyangfu/pytorch-vgg-cifar10
'''

import math

import torch.nn as nn
import torch.nn.init as init
from collections import OrderedDict

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


class VGG(nn.Module):
    '''
    VGG model
    '''

    def __init__(self, features):
        super(VGG, self).__init__()
        self.layer_outputs = OrderedDict()
        self.features = features
        self.classifier = nn.Sequential(OrderedDict([
            ('Dropout1', nn.Dropout()),
            ('fc6', nn.Linear(512, 512)),
            ('ReLu6', nn.ReLU(True)),
            ('Dropout2', nn.Dropout()),
            ('fc7', nn.Linear(512, 512)),
            ('ReLu7', nn.ReLU(True)),
            ('fc8', nn.Linear(512, 10)),])
        )
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def forward_and_save_outputs(self, x):
        self.layer_outputs = OrderedDict()
        for name, module in self.features.named_modules():
            if name != "":
                x = module(x)
            if 'conv' in name:
                self.layer_outputs.update([(name, x)])
        x = x.view(x.size(0), -1)
        for name, module in self.classifier.named_modules():
            if name != "":
                x = module(x)
            if 'fc' in name:
                self.layer_outputs.update([(name, x)])
        return x


def make_layers(cfg, batch_norm=False):
    layers = OrderedDict()
    in_channels = 3
    index0 = 1
    index1 = 1
    for v in cfg:
        if v == 'M':
            layers.update([('pool{}'.format(index0), nn.MaxPool2d(kernel_size=2, stride=2))])
            index0 += 1
            index1 = 1
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers.update([('conv{}_{}'.format(index0, index1), conv2d),
                               ('norm{}_{}'.format(index0, index1), nn.BatchNorm2d(v)),
                               ('relu{}_{}'.format(index0, index1), nn.ReLU(inplace=True))])
            else:
                layers.update([('conv{}_{}'.format(index0, index1), conv2d),
                           ('relu{}_{}'.format(index0, index1), nn.ReLU(inplace=True))])
            in_channels = v
            index1 += 1
    return nn.Sequential(layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
          512, 512, 512, 512, 'M'],
}


def vgg11():
    """VGG 11-layer model (configuration "A")"""
    return VGG(make_layers(cfg['A']))


def vgg11_bn():
    """VGG 11-layer model (configuration "A") with batch normalization"""
    return VGG(make_layers(cfg['A'], batch_norm=True))


def vgg13():
    """VGG 13-layer model (configuration "B")"""
    return VGG(make_layers(cfg['B']))


def vgg13_bn():
    """VGG 13-layer model (configuration "B") with batch normalization"""
    return VGG(make_layers(cfg['B'], batch_norm=True))


def vgg16():
    """VGG 16-layer model (configuration "D")"""
    return VGG(make_layers(cfg['D']))


def vgg16_bn():
    """VGG 16-layer model (configuration "D") with batch normalization"""
    return VGG(make_layers(cfg['D'], batch_norm=True))


def vgg19():
    """VGG 19-layer model (configuration "E")"""
    return VGG(make_layers(cfg['E']))


def vgg19_bn():
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    return VGG(make_layers(cfg['E'], batch_norm=True))
