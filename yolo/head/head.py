# -*- coding: utf-8 -*-
# Author: yohannxu
# Email: yuhannxu@gmail.com
# CreateTime: 2020-08-12 00:25:02
# Description: head.py

import os
import numpy as np
import pandas as pd
import cv2
from glob import glob
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers import ConvBnActivation
from ..utils import type_check
from easydict import EasyDict


class YOLOHead(nn.Module):

    @type_check(object, EasyDict)
    def __init__(self, cfg):
        super(YOLOHead, self).__init__()
        ac_type = cfg.HEAD.ACTIVATION_FN

        out_channel = (cfg.DATASET.NUM_CLASSES + 5) * 3

        self.conv1_1 = ConvBnActivation(128, 256, 3, 1, 1, ac_type)
        self.conv1_2 = nn.Conv2d(256, out_channel, 1, 1, 0)

        self.conv2_1 = ConvBnActivation(256, 512, 3, 1, 1, ac_type)
        self.conv2_2 = nn.Conv2d(512, out_channel, 1, 1, 0)

        self.conv3_1 = ConvBnActivation(512, 1024, 3, 1, 1, ac_type)
        self.conv3_2 = nn.Conv2d(1024, out_channel, 1, 1, 0)

    @type_check(object, torch.Tensor, torch.Tensor, torch.Tensor)
    def forward(self, x1, x2, x3):
        x1 = self.conv1_1(x1)
        x1 = self.conv1_2(x1)

        x2 = self.conv2_1(x2)
        x2 = self.conv2_2(x2)

        x3 = self.conv3_1(x3)
        x3 = self.conv3_2(x3)

        return [x1, x2, x3]
