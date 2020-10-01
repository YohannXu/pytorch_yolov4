# -*- coding: utf-8 -*-
# Author: yohannxu
# Email: yuhannxu@gmail.com
# CreateTime: 2020-08-19 13:48:55
# Description: lr.py

from bisect import bisect_right

import math
import torch.optim as optim


class WarmupMultiStepLR(optim.lr_scheduler._LRScheduler):
    """
    具有warmup的多步学习率下降策略
    """

    def __init__(
        self,
        optimizer,
        cfg,
        last_epoch=-1
    ):
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
            if self.warmup_method == 'constant':
                warmup_factor = self.warmup_factor
            elif self.warmup_method == 'linear':
                alpha = float(self.last_epoch) / self.warmup_iters
                warmup_factor = alpha
            elif self.warmup_method == 'power':
                alpha = float(self.last_epoch) / self.warmup_iters
                warmup_factor = pow(alpha, self.warmup_power)
        return [
            base_lr * warmup_factor * self.gamma ** bisect_right(
                self.milestones, self.last_epoch) for base_lr in self.base_lrs
        ]


class WarmupCosineLR(optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        num_iters,
        warmup_factor=1.0 / 3,
        warmup_iters=500,
        warmup_method='linear',
        gamma=0.99,
        last_epoch=-1
    ):
        self.num_iters = num_iters
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        self.gamma = gamma
        super(WarmupCosineLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == 'constant':
                warmup_factor = self.warmup_factor
            elif self.warmup_method == 'linear':
                alpha = float(self.last_epoch) / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        return [
                base_lr * (((1 + math.cos((self.last_epoch - self.warmup_iters) * math.pi / self.num_iters)) / 2) * self.gamma + 1 - self.gamma) for base_lr in self.base_lrs
                ]


class MultiStepLR(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, milestones, lrs, batch_size, last_epoch):
        self.milestones = milestones
        self.batch_size = batch_size
        self.lrs = [lr / self.batch_size for lr in lrs]
        super(MultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [self.lrs[bisect_right(self.milestones, self.last_epoch)] for base_lr in self.base_lrs]
