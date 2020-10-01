# -*- coding: utf-8 -*-
# Author: yohannxu
# Email: yuhannxu@gmail.com
# CreateTime: 2020-08-11 18:06:20
# Description: pan.py

import os
import numpy as np
import pandas as pd
import cv2
from glob import glob
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers import ConvBnActivation, Upsample
from ..utils import type_check
from easydict import EasyDict


class PAN(nn.Module):
    """
    先进行一次FPN, 再进行一次PAN
    """

    @type_check(object, EasyDict)
    def __init__(self, cfg):
        super(PAN, self).__init__()
        ac_type = cfg.PAN.ACTIVATION_FN
        # FPN
        self.conv1_1 = ConvBnActivation(2048, 512, 1, 1, 0, ac_type)
        self.conv1_2 = ConvBnActivation(512, 1024, 3, 1, 1, ac_type)
        self.conv1_3 = ConvBnActivation(1024, 512, 1, 1, 0, ac_type)
        self.up1 = Upsample()

        self.conv2_1 = ConvBnActivation(512, 256, 1, 1, 0, ac_type)
        self.conv2_2 = ConvBnActivation(512, 256, 1, 1, 0, ac_type)
        self.conv2_3 = ConvBnActivation(512, 256, 1, 1, 0, ac_type)
        self.conv2_4 = ConvBnActivation(256, 512, 3, 1, 1, ac_type)
        self.conv2_5 = ConvBnActivation(512, 256, 1, 1, 0, ac_type)
        self.conv2_6 = ConvBnActivation(256, 512, 3, 1, 1, ac_type)
        self.conv2_7 = ConvBnActivation(512, 256, 1, 1, 0, ac_type)
        self.up2 = Upsample()

        self.conv3_1 = ConvBnActivation(256, 128, 1, 1, 0, ac_type)
        self.conv3_2 = ConvBnActivation(256, 128, 1, 1, 0, ac_type)
        self.conv3_3 = ConvBnActivation(256, 128, 1, 1, 0, ac_type)
        self.conv3_4 = ConvBnActivation(128, 256, 3, 1, 1, ac_type)
        self.conv3_5 = ConvBnActivation(256, 128, 1, 1, 0, ac_type)
        self.conv3_6 = ConvBnActivation(128, 256, 3, 1, 1, ac_type)
        self.conv3_7 = ConvBnActivation(256, 128, 1, 1, 0, ac_type)

        # PAN
        self.conv4_1 = ConvBnActivation(128, 256, 3, 2, 1, ac_type)
        self.conv4_2 = ConvBnActivation(512, 256, 1, 1, 0, ac_type)
        self.conv4_3 = ConvBnActivation(256, 512, 3, 1, 1, ac_type)
        self.conv4_4 = ConvBnActivation(512, 256, 1, 1, 0, ac_type)
        self.conv4_5 = ConvBnActivation(256, 512, 3, 1, 1, ac_type)
        self.conv4_6 = ConvBnActivation(512, 256, 1, 1, 0, ac_type)

        self.conv5_1 = ConvBnActivation(256, 512, 3, 2, 1, ac_type)
        self.conv5_2 = ConvBnActivation(1024, 512, 1, 1, 0, ac_type)
        self.conv5_3 = ConvBnActivation(512, 1024, 3, 1, 1, ac_type)
        self.conv5_4 = ConvBnActivation(1024, 512, 1, 1, 0, ac_type)
        self.conv5_5 = ConvBnActivation(512, 1024, 3, 1, 1, ac_type)
        self.conv5_6 = ConvBnActivation(1024, 512, 1, 1, 0, ac_type)

    @type_check(object, torch.Tensor, torch.Tensor, torch.Tensor)
    def forward(self, x1, x2, x3):
        # FPN
        x = self.conv1_1(x1)
        x = self.conv1_2(x)
        x1 = self.conv1_3(x)

        x = self.conv2_1(x1)
        # 升采样后concat
        x = self.up1(x, x2)
        x2 = self.conv2_2(x2)
        x = torch.cat([x2, x], dim=1)
        x = self.conv2_3(x)
        x = self.conv2_4(x)
        x = self.conv2_5(x)
        x = self.conv2_6(x)
        x2 = self.conv2_7(x)

        x = self.conv3_1(x2)
        # 升采样后concat
        x = self.up2(x, x3)
        x3 = self.conv3_2(x3)
        x = torch.cat([x3, x], dim=1)
        x = self.conv3_3(x)
        x = self.conv3_4(x)
        x = self.conv3_5(x)
        x = self.conv3_6(x)
        x3 = self.conv3_7(x)

        # PAN
        # 降采样后concat
        x = self.conv4_1(x3)
        x = torch.cat([x, x2], dim=1)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        x = self.conv4_4(x)
        x = self.conv4_5(x)
        x2 = self.conv4_6(x)

        # 降采样后concat
        x = self.conv5_1(x2)
        x = torch.cat([x, x1], dim=1)
        x = self.conv5_2(x)
        x = self.conv5_3(x)
        x = self.conv5_4(x)
        x = self.conv5_5(x)
        x1 = self.conv5_6(x)

        return [x3, x2, x1]
