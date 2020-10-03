# -*- coding: utf-8 -*-
# Author: yohannxu
# Email: yuhannxu@gmail.com
# CreateTime: 2020-08-19 13:48:55
# Description: 学习率下降策略

import math
from bisect import bisect_right

import torch.optim as optim
from easydict import EasyDict

from yolo.utils import type_check


class WarmupMultiStepLR(optim.lr_scheduler._LRScheduler):
    """
    具有warmup的多步学习率下降策略
    """

    @type_check(object, optim.Optimizer, EasyDict, int)
    def __init__(self, optimizer, cfg, last_epoch=-1):
        """
        Args:
            optimizer: 优化器
            cfg: 配置文件
            last_epoch: 最后迭代次数
        """

        self.milestones = cfg.OPTIMIZER.STEPS
        self.gamma = cfg.OPTIMIZER.GAMMA
        self.warmup_factor = cfg.OPTIMIZER.WARMUP_FACTOR
        self.warmup_iters = cfg.OPTIMIZER.WARMUP_ITERS
        self.warmup_method = cfg.OPTIMIZER.WARMUP_METHOD
        self.warmup_power = cfg.OPTIMIZER.WARMUP_POWER
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            # 常量
            if self.warmup_method == 'constant':
                warmup_factor = self.warmup_factor
            # 线性递增
            elif self.warmup_method == 'linear':
                alpha = float(self.last_epoch) / self.warmup_iters
                warmup_factor = alpha
            # 指数递增
            elif self.warmup_method == 'power':
                alpha = float(self.last_epoch) / self.warmup_iters
                warmup_factor = pow(alpha, self.warmup_power)
        return [
            base_lr * warmup_factor * self.gamma ** bisect_right(
                self.milestones, self.last_epoch) for base_lr in self.base_lrs
        ]


class WarmupCosineLR(optim.lr_scheduler._LRScheduler):
    @type_check(object, optim.Optimizer, EasyDict, int)
    def __init__(self, optimizer, cfg, last_epoch=-1):
        """
        Args:
            optimizer: 优化器
            cfg: 配置文件
            last_epoch: 最后迭代次数
        """

        self.num_iters = cfg.TRAIN.NUM_BATCHES
        self.warmup_factor = cfg.OPTIMIZER.WARMUP_FACTOR
        self.warmup_iters = cfg.OPTIMIZER.WARMUP_ITERS
        self.warmup_method = cfg.OPTIMIZER.WARMUP_METHOD
        self.gamma = cfg.OPTIMIZER.COSINE_GAMMA
        super(WarmupCosineLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == 'constant':
                warmup_factor = self.warmup_factor
            elif self.warmup_method == 'linear':
                alpha = float(self.last_epoch) / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
            elif self.warmup_method == 'power':
                alpha = float(self.last_epoch) / self.warmup_iters
                warmup_factor = pow(alpha, self.warmup_power)
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        return [
            base_lr * (((1 + math.cos((self.last_epoch - self.warmup_iters) * math.pi / self.num_iters)) / 2) * self.gamma + 1 - self.gamma) for base_lr in self.base_lrs
        ]
