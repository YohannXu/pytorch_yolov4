# -*- coding: utf-8 -*-
# Author: yohannxu
# Email: yuhannxu@gmail.com
# CreateTime: 2020-08-11 17:42:31
# Description: 空间金字塔池化

import torch
import torch.nn as nn
from easydict import EasyDict

from ..layers import ConvBnActivation
from ..utils import type_check


class SPP(nn.Module):
    """
    空间金字塔池化
    """

    @type_check(object, EasyDict)
    def __init__(self, cfg):
        """
        Args:
            cfg: 配置文件
        """

        super(SPP, self).__init__()
        ac_type = cfg.SPP.ACTIVATION_FN
        self.conv1 = ConvBnActivation(1024, 512, 1, 1, 0, ac_type)
        self.conv2 = ConvBnActivation(512, 1024, 3, 1, 1, ac_type)
        self.conv3 = ConvBnActivation(1024, 512, 1, 1, 0, ac_type)
        self.pool1 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=9, stride=1, padding=4)
        self.pool3 = nn.MaxPool2d(kernel_size=13, stride=1, padding=6)

    @type_check(object, torch.Tensor, torch.Tensor, torch.Tensor)
    def forward(self, x1, x2, x3):
        """
        对尺寸最低的features进行三次池化, 对池化结果进行concat
        """

        x = self.conv1(x1)
        x = self.conv2(x)
        x = self.conv3(x)
        p1 = self.pool1(x)
        p2 = self.pool2(x)
        p3 = self.pool3(x)
        spp = torch.cat([p3, p2, p1, x], dim=1)

        return [spp, x2, x3]
