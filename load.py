# -*- coding: utf-8 -*-
# Author: yohannxu
# Email: yuhannxu@gmail.com
# CreateTime: 2020-08-19 18:23:28
# Description: load.py

import os
import numpy as np
import pandas as pd
import cv2
from glob import glob
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F


from model import Model
from default import cfg

model = Model(cfg, pretrained=False)

with open('yolov4.weights', 'rb') as f:
    header = np.fromfile(f, count=5, dtype=np.int32)
    weights = np.fromfile(f, dtype=np.float32)

    start = 0
    conv = None
    index = 0
    preserve_weights = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            if index == 552:
                preserve_weights.append(weights[start: start + 361471])
                start += 361471
            elif index == 588:
                preserve_weights.append(weights[start: start + 1312511])
                start += 1312511
            elif index == 624:
                preserve_weights.append(weights[start: start + 4984063])
                start += 4984063
                break

            if conv is None:
                conv = m
            else:
                num = conv.weight.numel()
                conv.weight.data.copy_(torch.from_numpy(weights[start: start + num]).view_as(conv.weight))
                start += num
                index += 1
                try:
                    num = conv.bias.numel()
                    conv.bias.data.copy_(torch.from_numpy(weights[start: start + num]))
                    start += num
                    index += 1
                except Exception:
                    pass
                conv = m
        elif isinstance(m, nn.BatchNorm2d):
            num = m.bias.numel()
            m.bias.data.copy_(torch.from_numpy(weights[start: start + num]))
            start += num
            index += 1
            m.weight.data.copy_(torch.from_numpy(weights[start: start + num]))
            start += num
            index += 1
            m.running_mean.data.copy_(torch.from_numpy(weights[start: start + num]))
            start += num
            index += 1
            m.running_var.data.copy_(torch.from_numpy(weights[start: start + num]))
            start += num
            index += 2

            num = conv.weight.numel()
            conv.weight.data.copy_(torch.from_numpy(weights[start: start + num]).view_as(conv.weight))
            start += num
            index += 1
            try:
                num = conv.bias.numel()
                conv.bias.data.copy_(torch.from_numpy(weights[start: start + num]))
                start += num
                index += 1
            except Exception:
                pass
            conv = None
    if conv:
        num = conv.weight.numel()
        conv.weight.data.copy_(torch.from_numpy(weights[start: start + num]).view_as(conv.weight))
        start += num
        index += 1
        try:
            num = conv.bias.numel()
            conv.bias.data.copy_(torch.from_numpy(weights[start: start + num]))
            start += num
            index += 1
        except Exception:
            pass
        conv = None
    print(index)

    for k, v in model.state_dict().items():
        if k == 'head.conv1_1.conv.weight':
            v.data.copy_(torch.from_numpy(preserve_weights[0][1024: 295936]).view_as(v))
        if k == 'head.conv1_1.bn.weight':
            v.data.copy_(torch.from_numpy(preserve_weights[0][256: 512]))
        if k == 'head.conv1_1.bn.bias':
            v.data.copy_(torch.from_numpy(preserve_weights[0][0: 256]))
        if k == 'head.conv1_1.bn.running_mean':
            v.data.copy_(torch.from_numpy(preserve_weights[0][512: 768]))
        if k == 'head.conv1_1.bn.running_var':
            v.data.copy_(torch.from_numpy(preserve_weights[0][768: 1024]))
        if k == 'head.conv1_2.weight':
            v.data.copy_(torch.from_numpy(preserve_weights[0][296191: 361471]).view_as(v))
        if k == 'head.conv1_2.bias':
            v.data.copy_(torch.from_numpy(preserve_weights[0][295936: 296191]))

        if k == 'head.conv2_1.conv.weight':
            v.data.copy_(torch.from_numpy(preserve_weights[1][2048: 1181696]).view_as(v))
        if k == 'head.conv2_1.bn.weight':
            v.data.copy_(torch.from_numpy(preserve_weights[1][512: 1024]))
        if k == 'head.conv2_1.bn.bias':
            v.data.copy_(torch.from_numpy(preserve_weights[1][0: 512]))
        if k == 'head.conv2_1.bn.running_mean':
            v.data.copy_(torch.from_numpy(preserve_weights[1][1024: 1536]))
        if k == 'head.conv2_1.bn.running_var':
            v.data.copy_(torch.from_numpy(preserve_weights[1][1536: 2048]))
        if k == 'head.conv2_2.weight':
            v.data.copy_(torch.from_numpy(preserve_weights[1][1181951: 1312511]).view_as(v))
        if k == 'head.conv2_2.bias':
            v.data.copy_(torch.from_numpy(preserve_weights[1][1181696: 1181951]))

        if k == 'head.conv3_1.conv.weight':
            v.data.copy_(torch.from_numpy(preserve_weights[2][4096: 4722688]).view_as(v))
        if k == 'head.conv3_1.bn.weight':
            v.data.copy_(torch.from_numpy(preserve_weights[2][1024: 2048]))
        if k == 'head.conv3_1.bn.bias':
            v.data.copy_(torch.from_numpy(preserve_weights[2][0: 1024]))
        if k == 'head.conv3_1.bn.running_mean':
            v.data.copy_(torch.from_numpy(preserve_weights[2][2048: 3072]))
        if k == 'head.conv3_1.bn.running_var':
            v.data.copy_(torch.from_numpy(preserve_weights[2][3072: 4096]))
        if k == 'head.conv3_2.weight':
            v.data.copy_(torch.from_numpy(preserve_weights[2][4722943: 4984063]).view_as(v))
        if k == 'head.conv3_2.bias':
            v.data.copy_(torch.from_numpy(preserve_weights[2][4722688: 4722943]))


for i, (n, v) in enumerate(model.state_dict().items()):
    if len(v.shape):
        print(i, n, v.shape, v.mean(), v.std())

model = model.cuda()
torch.save({'state_dict': model.state_dict()}, 'yolov4.pth')
