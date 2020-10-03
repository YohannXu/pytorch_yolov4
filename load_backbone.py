# -*- coding: utf-8 -*-
# Author: yohannxu
# Email: yuhannxu@gmail.com
# CreateTime: 2020-08-19 23:23:11
# Description: load_backbone.py

import numpy as np
import torch
import torch.nn as nn

from default import cfg
from model import Model

model = Model(cfg, pretrained=False)

with open('csdarknet53-omega.conv.105', 'rb') as f:
    header = np.fromfile(f, count=5, dtype=np.int32)
    weights = np.fromfile(f, dtype=np.float32)

    start = 0
    for m in model.backbone.modules():
        if isinstance(m, nn.Conv2d):
            conv = m
        elif isinstance(m, nn.BatchNorm2d):
            num = m.bias.numel()
            m.bias.data.copy_(torch.from_numpy(weights[start: start + num]))
            start += num
            m.weight.data.copy_(torch.from_numpy(weights[start: start + num]))
            start += num
            m.running_mean.data.copy_(torch.from_numpy(weights[start: start + num]))
            start += num
            m.running_var.data.copy_(torch.from_numpy(weights[start: start + num]))
            start += num

            num = conv.weight.numel()
            conv.weight.data.copy_(torch.from_numpy(weights[start: start + num]).view_as(conv.weight))
            start += num

torch.save({'state_dict': model.backbone.state_dict()}, 'yolov4_backbone.pth')
