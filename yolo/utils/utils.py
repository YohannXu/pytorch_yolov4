# -*- coding: utf-8 -*-
# Author: yohannxu
# Email: yuhannxu@gmail.com
# CreateTime: 2020-08-10 17:23:41
# Description: utils.py

import os
from collections import deque
from functools import wraps
from glob import glob
from inspect import signature

import torch


def type_check(*types):
    """
    用于输入参数类型检查
    Args:
        *types: 目标函数的参数类型
    """
    def decorate(func):
        sig = signature(func)
        # 获取目标函数的参数名和参数类型
        # {name: type}
        arg_types = sig.bind_partial(*types).arguments
        params = sig.parameters

        @wraps(func)
        def wrapper(*args, **kwargs):
            # 获取目标函数输入参数
            # {name: value}
            input_args = sig.bind(*args, **kwargs)
            for name, value in input_args.arguments.items():
                if name in arg_types:
                    if not isinstance(value, arg_types[name]):
                        if params[name].KEYWORD_ONLY and params[name].default == value:
                            pass
                        else:
                            raise TypeError('Argument {} must be {}'.format(name, arg_types[name]))
            return func(*args, **kwargs)
        return wrapper
    return decorate


model_urls = {
    50: 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    101: 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    152: 'https://download.pytorch.org/models/resnet152-b121ed2d.pth'
}


@type_check(int)
def load_state_dict_from_url(num_layers):
    weights = {}
    weight = torch.hub.load_state_dict_from_url(model_urls[num_layers])

    for name, value in weight.items():
        if name == 'conv1.weight':
            name = 'stem.conv.weight'
        if name == 'bn1.weight':
            name = 'stem.bn.weight'
        if name == 'bn1.bias':
            name = 'stem.bn.bias'
        if name == 'bn1.running_mean':
            name = 'stem.bn.running_mean'
        if name == 'bn1.running_var':
            name = 'stem.bn.running_var'
        weight.requires_grad = True
        weights[name] = value

    return weights


def last_checkpoint(cfg):
    """
    返回最新权重文件名
    """
    checkpoints = glob('{}/*.pth'.format(cfg.OUTPUT))
    final_model = '{}/model_final.pth'.format(cfg.OUTPUT)
    if final_model in checkpoints:
        return final_model
    if checkpoints:
        checkpoints = sorted(checkpoints, key=lambda x: int(
            os.path.basename(x).split('.')[0].split('_')[-1]
        ))
        return checkpoints[-1]
    return checkpoints


class SmoothAverage():
    @type_check(object, int)
    def __init__(self, window_size=100):
        self.deque = deque(maxlen=window_size)
        self.val = 0
        self.sum = 0
        self.count = 0

    @type_check(object, float)
    def update(self, val):
        self.val = val
        self.deque.append(val)
        self.sum += val
        self.count += 1

    @property
    def avg(self):
        return torch.tensor(list(self.deque)).mean().item()

    @property
    def global_avg(self):
        return self.sum / self.count


class Metric(dict):
    def __init__(self, metric=SmoothAverage, delimiter='    '):
        self['iou_loss'] = metric()
        self['obj_loss'] = metric()
        self['cls_loss'] = metric()
        self['loss'] = metric()
        self['time'] = metric()
        self.delimiter = delimiter

    def update(self, k, v):
        if k not in self:
            raise KeyError('{}'.format(k))
        if isinstance(v, torch.Tensor):
            v = v.item()
        self[k].update(v)

    def __str__(self):
        loss_str = []
        for name, meter in self.items():
            loss_str.append(
                '{}: {:.4f} ({:.4f})'.format(name, meter.val, meter.avg)
            )
        return self.delimiter.join(loss_str)
