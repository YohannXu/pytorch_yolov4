# -*- coding: utf-8 -*-
# Author: yohannxu
# Email: yuhannxu@gmail.com
# CreateTime: 2020-08-19 04:18:10
# Description: models.py

import torch
import torch.nn as nn
from apex import amp

from yolo.backbone import CSPDarkNet53
from yolo.head import YOLOHead
from yolo.neck import PAN, SPP
from yolo.yolo import YOLO, OnnxYOLO


class Model(nn.Module):
    def __init__(self, cfg, pretrained=True):
        super(Model, self).__init__()
        self.backbone = CSPDarkNet53(cfg)
        self.spp = SPP(cfg)
        self.pan = PAN(cfg)
        self.head = YOLOHead(cfg)
        self.yolo = YOLO(cfg)

        if pretrained:
            print('loading imagenet pretrained weights!')
            self.backbone.load_state_dict(torch.load(cfg.TRAIN.PRETRAIN_WEIGHT)['state_dict'])

        if cfg.TRAIN.MIX_LEVEL == 'O1':
            self.yolo.loss.forward = amp.half_function(self.yolo.loss.forward)

    def forward(self, x, targets=None, cats=None):
        x = self.backbone(x)
        x = self.spp(*x)
        x = self.pan(*x)
        x = self.head(*x)
        x = self.yolo(x, targets, cats)

        return x


class OnnxModel(nn.Module):
    def __init__(self, cfg):
        super(OnnxModel, self).__init__()
        self.backbone = CSPDarkNet53(cfg)
        self.spp = SPP(cfg)
        self.pan = PAN(cfg)
        self.head = YOLOHead(cfg)
        self.yolo = OnnxYOLO(cfg)

    def forward(self, x):
        x = self.backbone(x)
        x = self.spp(*x)
        x = self.pan(*x)
        x = self.head(*x)
        x = self.yolo(x)

        return x
