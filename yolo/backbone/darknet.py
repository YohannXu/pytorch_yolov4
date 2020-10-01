# -*- coding: utf-8 -*-
# Author: yohannxu
# Email: yuhannxu@gmail.com
# CreateTime: 2020-08-10 17:24:07
# Description: darknet.py

import torch
import torch.nn as nn

from ..layers import ConvBnActivation
from ..utils import type_check
from easydict import EasyDict


class ShortcutBlock(nn.Module):
    """
    残差连接网络结构
    input ---
      |     |
      |   conv1
      |     |
      |   conv2
      |     |
      +------
      |
      output
    """

    @type_check(object, int, str)
    def __init__(self, index, ac_type):
        super(ShortcutBlock, self).__init__()
        if index == 1:
            self.conv1 = ConvBnActivation(64, 32, 1, 1, 0, ac_type)
            self.conv2 = ConvBnActivation(32, 64, 3, 1, 1, ac_type)
        else:
            channel = 32 * 2 ** (index - 1)
            self.conv1 = ConvBnActivation(channel, channel, 1, 1, 0, ac_type)
            self.conv2 = ConvBnActivation(channel, channel, 3, 1, 1, ac_type)

    @type_check(object, torch.Tensor)
    def forward(self, x):
        shortcut = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + shortcut

        return x


class LayerBlock(nn.Module):
    """
    backbone每一层的网络结构
    input
      |
    conv1 ----
      |      |
    conv2  conv3
      |      |
      |   shortcut
      |      |
      |    conv4
      |      |
    concat ---
      |
    conv5
      |
    output
    """

    @type_check(object, list, int, str)
    def __init__(self, num_layers, index, ac_type):
        super(LayerBlock, self).__init__()
        channel = 32 * 2 ** (index - 1)

        if index == 1:
            self.conv1 = ConvBnActivation(32, 64, 3, 2, 1, ac_type)
            self.conv2 = ConvBnActivation(64, 64, 1, 1, 0, ac_type)
            self.conv3 = ConvBnActivation(64, 64, 1, 1, 0, ac_type)
        else:
            self.conv1 = ConvBnActivation(channel, channel * 2, 3, 2, 1, ac_type)
            self.conv2 = ConvBnActivation(channel * 2, channel, 1, 1, 0, ac_type)
            self.conv3 = ConvBnActivation(channel * 2, channel, 1, 1, 0, ac_type)

        modules = []
        for _ in range(num_layers[index - 1]):
            modules.append(ShortcutBlock(index, ac_type))
        self.shortcut = nn.Sequential(*modules)

        if index == 1:
            self.conv4 = ConvBnActivation(64, 64, 1, 1, 0, ac_type)
            self.conv5 = ConvBnActivation(64 * 2, 64, 1, 1, 0, ac_type)
        else:
            self.conv4 = ConvBnActivation(channel, channel, 1, 1, 0, ac_type)
            self.conv5 = ConvBnActivation(channel * 2, channel * 2, 1, 1, 0, ac_type)

    @type_check(object, torch.Tensor)
    def forward(self, x):

        x = self.conv1(x)
        x1 = self.conv2(x)
        x = self.conv3(x)
        x = self.shortcut(x)
        x2 = self.conv4(x)
        x3 = torch.cat([x2, x1], dim=1)
        x = self.conv5(x3)

        return x


class CSPDarkNet53(nn.Module):

    @type_check(object, EasyDict)
    def __init__(self, cfg):
        super(CSPDarkNet53, self).__init__()
        ac_type = cfg.BACKBONE.ACTIVATION_FN
        num_layers = [1, 2, 8, 8, 4]

        self.stem = ConvBnActivation(3, 32, 3, 1, 1, ac_type)
        self.layer1 = LayerBlock(num_layers, 1, ac_type)
        self.layer2 = LayerBlock(num_layers, 2, ac_type)
        self.layer3 = LayerBlock(num_layers, 3, ac_type)
        self.layer4 = LayerBlock(num_layers, 4, ac_type)
        self.layer5 = LayerBlock(num_layers, 5, ac_type)

    @type_check(object, torch.Tensor)
    def forward(self, x):
        x = self.stem(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)

        return [x5, x4, x3]
