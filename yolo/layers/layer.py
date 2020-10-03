# -*- coding: utf-8 -*-
# Author: yohannxu
# Email: yuhannxu@gmail.com
# CreateTime: 2020-08-11 17:38:46
# Description: layer定义

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import type_check


class Mish(nn.Module):
    """
    https://arxiv.org/abs/1908.08681
    """

    def __init__(self):
        super(Mish, self).__init__()

    @type_check(object, torch.Tensor)
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class ConvBnActivation(nn.Module):
    """
    卷积+BN+激活函数
    """

    @type_check(object, int, int, int, int, int, str, bool)
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1, ac_type=None, bias=False):
        """
        Args:
            in_channel: 输入通道数
            out_channel: 输出通道数
            kernel_size: 卷积核大小
            stride: 卷积步长
            padding: 卷积填充大小
            ac_type: 激活函数类型
            bias: 是否使用偏置
        """

        super(ConvBnActivation, self).__init__()

        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channel)
        if ac_type == 'mish':
            self.ac_fn = Mish()
        elif ac_type == 'relu':
            self.ac_fn = nn.ReLU(inplace=True)
        elif ac_type == 'leaky':
            self.ac_fn = nn.LeakyReLU(0.1, inplace=True)

    @type_check(object, torch.Tensor)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.ac_fn(x)
        return x


class Upsample(nn.Module):
    """
    升采样层
    """

    def __init__(self):
        super(Upsample, self).__init__()

    @type_check(object, torch.Tensor, torch.Tensor)
    def forward(self, x, target):
        th, tw = target.shape[2:]

        if self.training:
            return F.interpolate(x, size=(th, tw), mode='nearest')
        b, c, h, w = x.shape
        return x.view(b, c, h, 1, w, 1).expand(b, c, h, th // h, w, tw // w).contiguous().view(b, c, th, tw)
