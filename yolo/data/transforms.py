# -*- coding: utf-8 -*-
# Author: yohannxu
# Email: yuhannxu@gmail.com
# CreateTime: 2020-08-17 17:02:38
# Description: 各种数据预处理

import math
import random

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F
from easydict import EasyDict
from PIL import Image
from torchvision import transforms as T

from ..utils import type_check


@type_check(EasyDict, bool)
def build_transforms(cfg, is_train=True):
    """
    数据预处理
    Args:
        cfg: 配置文件
        is_train: 是否处于训练模式
    """

    if is_train:
        transforms = T.Compose(
            [
                MultiSizes(cfg),
                Mosaic(cfg),
                ColorJitter(cfg),
                RandomEraser(cfg),
                Rotate(cfg),
                RandomHorizontalFlip(cfg),
                Resize(cfg, is_train=True),
                ToTensor(),
            ]
        )
    else:
        transforms = T.Compose(
            [
                Resize(cfg, is_train=False),
                ToTensor(),
            ]
        )

    return transforms


class MultiSizes():
    """
    多尺寸训练
    """

    @type_check(object, EasyDict)
    def __init__(self, cfg):
        """
        Args:
            cfg: 配置文件
        """

        self.cfg = cfg
        self.sizes = cfg.DATASET.SIZES
        self.frequency = cfg.DATASET.FREQUENCY
        self.time_to_change = self.frequency

    @type_check(object, dict)
    def __call__(self, inputs):
        """
        每隔frequency次改变图片缩放尺寸
        Args:
            inputs: 输入数据
        """

        self.time_to_change -= 1
        if self.time_to_change == 0:
            self.cfg.DATASET.SIZE = random.choice(self.sizes)
            self.time_to_change = self.frequency
        return inputs


class Mosaic():
    """
    将4张图片拼接在一起
    """

    @type_check(object, EasyDict, int)
    def __init__(self, cfg, interpolation=Image.BILINEAR):
        """
        Args:
            cfg: 配置文件
            interpolation: 插值算法
        """

        self.resize_range = cfg.DATASET.MOSAIC_RESIZE_RANGE
        self.color = cfg.DATASET.MOSAIC_COLOR
        self.size = cfg.DATASET.SIZE

    @type_check(object, dict)
    def __call__(self, inputs):
        """
        Args:
            inputs: 输入数据
        """

        images = inputs['image']
        bboxes = inputs['bbox']
        cats = inputs['cat']

        # 如果images不是列表格式, 表明未启用mosaic数据增强
        if type(images) is not list:
            return inputs

        # 空白图片, 用于拼接
        new_image = np.full((self.size, self.size, 3), self.color, dtype=np.uint8)
        new_bboxes = []
        new_cats = []
        # 生成随机中心点
        cx, cy = [random.randint(self.size // 4, self.size // 4 * 3) for _ in range(2)]
        # 生成随机缩放因子
        resize_ratio = random.uniform(self.resize_range[0], self.resize_range[1])
        # 计算得到缩放后的尺寸
        size = int(self.size * resize_ratio)

        for index, (image, bbox, cat) in enumerate(zip(images, bboxes, cats)):
            w, h = image.shape[:2]

            # 对每张图片根据缩放后的尺寸进行缩放
            if w > h:
                resize_h = int(size / w * h)
                image = cv2.resize(image, (resize_h, size))
                ratio_w = size / w
                ratio_h = resize_h / h
            else:
                resize_w = int(size / h * w)
                image = cv2.resize(image, (size, resize_w))
                ratio_w = resize_w / w
                ratio_h = size / h

            # 对bbox的宽高乘上对应缩放因子
            bbox[:, 0::2] *= ratio_w
            bbox[:, 1::2] *= ratio_h

            # 计算4张图片的拼接位置
            h, w = image.shape[:2]
            if index == 0:
                x1a, y1a, x2a, y2a = max(cx - w, 0), max(cy - h, 0), cx, cy
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif index == 1:
                x1a, y1a, x2a, y2a = cx, max(cy - h, 0), min(cx + w, self.size), cy
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif index == 2:
                x1a, y1a, x2a, y2a = max(cx - w, 0), cy, cx, min(cy + h, self.size)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(cx, w), min(y2a - y1a, h)
            elif index == 3:
                x1a, y1a, x2a, y2a = cx, cy, min(cx + w, self.size), min(cy + h, self.size)
                x1b, y1b, x2b, y2b = 0, 0, min(x2a - x1a, w), min(y2a - y1a, h)
            new_image[y1a: y2a, x1a: x2a] = image[y1b: y2b, x1b: x2b]

            # 得到每张图片bbox的位移
            offset_w = x1a - x1b
            offset_h = y1a - y1b
            # bbox加上位移
            bbox[:, 0::2] += offset_w
            bbox[:, 1::2] += offset_h

            # 得到位于图片外的bbox bool mask
            out_mask = (bbox[:, [0, 2]] <= 0).all(1) | (bbox[:, [0, 2]] >= self.size).all(
                1) | (bbox[:, [1, 3]] <= 0).all(1) | (bbox[:, [1, 3]] >= self.size).all(1)
            # 得到位于图片内的bbox bool mask
            mask = ~out_mask
            # 得到对应bbox和类别索引
            # TODO 此处clamp的max值有极小概率会导致报错, 可能减去一个较小值
            bbox = bbox[mask].clamp(0, self.size)
            cat = cat[mask]
            new_bboxes.append(bbox)
            new_cats.append(cat)
        bboxes = torch.cat(new_bboxes, dim=0)
        cats = torch.cat(new_cats, dim=0)

        inputs['image'] = Image.fromarray(new_image)
        inputs['bbox'] = bboxes
        inputs['cat'] = cats

        return inputs


class Resize():
    """
    图片缩放
    """

    @type_check(object, EasyDict, bool, int)
    def __init__(self, cfg, is_train=True, interpolation=Image.BILINEAR):
        """
        Args:
            cfg, 配置文件
            interpolation: 插值算法
        """

        if is_train:
            self.size = cfg.DATASET.SIZE
        else:
            self.size = cfg.DATASET.TEST_SIZE
        self.color = cfg.DATASET.RESIZE_COLOR
        self.interpolation = interpolation

    @type_check(object, dict)
    def __call__(self, inputs):
        """
        Args:
            inputs: 输入数据
        """

        image = inputs['image']

        w, h = image.size

        if w > h:
            resize_h = int(self.size / w * h)
            image = image.resize((self.size, resize_h))
            ratio_w = self.size / w
            ratio_h = resize_h / h
            offset_w = 0
            offset_h = random.randint(0, self.size - resize_h)
        else:
            resize_w = int(self.size / h * w)
            image = image.resize((resize_w, self.size))
            ratio_w = resize_w / w
            ratio_h = self.size / h
            offset_w = random.randint(0, self.size - resize_w)
            offset_h = 0

        # 在图片四周进行填充, 使其尺寸符合网络输入要求
        new_image = Image.new('RGB', (self.size, self.size), tuple(self.color))
        new_image.paste(image, (offset_w, offset_h))

        inputs['image'] = new_image
        inputs['ratio'] = [ratio_w, ratio_h]
        inputs['offset'] = [offset_w, offset_h]

        # 根据填充情况调整bbox坐标
        if 'bbox' in inputs:
            bbox = inputs['bbox']
            bbox[:, 0::2] *= ratio_w
            bbox[:, 1::2] *= ratio_h
            bbox[:, 0::2] += offset_w
            bbox[:, 1::2] += offset_h
            inputs['bbox'] = bbox

        return inputs


class RandomHorizontalFlip():
    """
    随机水平翻转
    """

    @type_check(object, EasyDict)
    def __init__(self, cfg):
        """
        Args:
            cfg: 配置文件
        """

        self.prob = cfg.DATASET.FLIP_HORIZONTAL_PROB

    @type_check(object, dict)
    def __call__(self, inputs):
        """
        Args:
            inputs: 输入数据
        """

        image = inputs['image']
        bbox = inputs['bbox']

        if random.random() < self.prob:
            # 翻转图片
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            w, h = image.size
            # 翻转bbox
            bbox[:, 0] = w - bbox[:, 0]
            bbox[:, 2] = w - bbox[:, 2]
            bbox[:, [0, 2]] = bbox[:, [2, 0]]

        inputs['image'] = image
        inputs['bbox'] = bbox

        return inputs


class ColorJitter():
    """
    随机色彩调整
    """

    @type_check(object, EasyDict)
    def __init__(self, cfg):
        """
        Args:
            cfg: 配置文件
        """

        brightness = cfg.DATASET.BRIGHTNESS
        contrast = cfg.DATASET.CONTRAST
        saturation = cfg.DATASET.SATURATION
        hue = cfg.DATASET.HUE

        self.color_jitter = T.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue
        )

    @type_check(object, dict)
    def __call__(self, inputs):
        """
        Args:
            inputs: 输入数据
        """

        image = inputs['image']
        image = self.color_jitter(image)
        inputs['image'] = image

        return inputs


class RandomEraser():
    """
    对每一个box进行随机擦除
    如果擦除区域太大，可能会遮盖目标的重要部分
    如果擦除区域太小，可能不会起太大作用
    """

    @type_check(object, EasyDict)
    def __init__(self, cfg):
        """
        Args:
            cfg: 配置文件
        """

        self.size_ratio = cfg.DATASET.ERASER_SIZE_RATIO
        self.prob = cfg.DATASET.ERASER_PROB
        self.color = cfg.DATASET.ERASER_COLOR

    @type_check(object, dict)
    def __call__(self, inputs):
        """
        在每个bbox范围内, 随机进行小范围的擦除
        Args:
            inputs: 输入数据
        """

        image = inputs['image']
        bbox = inputs['bbox']

        for box in bbox:
            if random.random() > self.prob:
                w = box[2] - box[0]
                h = box[3] - box[1]
                # 擦除区域的宽高
                eraser_w = random.randint(int(w * self.size_ratio[0]), int(w * self.size_ratio[1]))
                eraser_h = random.randint(int(h * self.size_ratio[0]), int(h * self.size_ratio[1]))
                # 擦除坐标
                eraser_x = random.randint(int(box[0]), int(box[2] - eraser_w))
                eraser_y = random.randint(int(box[1]), int(box[3] - eraser_h))
                rec = Image.new('RGB', (eraser_w, eraser_h), tuple(self.color))
                image.paste(rec, (eraser_x, eraser_y))

        inputs['image'] = image
        return inputs


class GridMask():
    """
    https://arxiv.org/abs/2001.04086
    """
    @type_check(object, list, list, list, int)
    def __init__(self, r, d, probs, max_iters):
        self.r = r
        self.d = d
        self.probs = probs
        self.max_iters = max_iters
        self.iter = 1

    @type_check(object, dict)
    def __call__(self, inputs):
        cur_prob = self.prob[0] * (1 - self.iter / self.max_iters) + self.prob[1] * self.iter / self.max_iters
        self.iter += 1

        if random.random() < cur_prob:
            image = inputs['image']
            h, w = image.shape[:2]

            d_w_min = int(w * self.d[0])
            d_w_max = int(w * self.d[1])
            d_h_min = int(h * self.d[0])
            d_h_max = int(h * self.d[1])

            dx = random.randint(d_w_min, d_w_max)
            dy = random.randint(d_h_min, d_h_max)

            l_w_min = int(dx * self.r[0])
            l_w_max = int(dx * self.r[1])
            l_h_min = int(dy * self.r[0])
            l_h_max = int(dy * self.r[1])

            lx = random.randint(l_w_min, l_w_max)
            ly = random.randint(l_h_min, l_h_max)

            sx = random.randint(0, dx)
            sy = random.randint(0, dy)

            hh = int(1.5 * h)
            ww = int(1.5 * w)
            mask = np.ones((hh, ww), np.float32)

            for i in range(ww / dx):
                s = dx * i + sx
                e = min(s + lx, ww - 1)
                mask[:, s: e] = 0

            for i in range(hh / dy):
                s = dy * i + sy
                e = min(s + ly, hh - 1)
                mask[s: e, :] = 0

            x = random.randint(0, ww - w - 1)
            y = random.randint(0, hh - h - 1)
            mask = mask[y: y + h, x: x + w]
            mask = 1 - mask

            image_np = np.array(image)
            image_np *= mask
            image = Image.fromarray(image_np)

            inputs['image'] = image

        return inputs


class Rotate():
    """
    旋转图片
    """

    @type_check(object, EasyDict)
    def __init__(self, cfg):
        """
        Args:
            cfg: 配置文件
        """

        self.range = cfg.DATASET.ROTATE_RANGE
        self.prob = cfg.DATASET.ROTATE_PROB

    @type_check(object, dict)
    def __call__(self, inputs):
        """
        Args:
            inputs: 输入数据
        """

        if random.random() > self.prob:
            image = inputs['image']
            w, h = image.size
            bbox = inputs['bbox']

            # 得到随机角度
            angle = random.randint(self.range[0], self.range[1])
            # 旋转图片
            image = image.rotate(angle, expand=True)

            # 旋转bbox
            sin = math.sin(math.pi / 180 * abs(angle))
            cos = math.cos(math.pi / 180 * abs(angle))
            new_bbox = torch.zeros_like(bbox)

            if angle < 0:
                new_bbox[:, 0] = bbox[:, 0] * cos - bbox[:, 3] * sin + h * sin
                new_bbox[:, 1] = bbox[:, 0] * sin + bbox[:, 1] * cos
                new_bbox[:, 2] = bbox[:, 2] * cos - bbox[:, 1] * sin + h * sin
                new_bbox[:, 3] = bbox[:, 2] * sin + bbox[:, 3] * cos
            else:
                new_bbox[:, 0] = bbox[:, 0] * cos + bbox[:, 1] * sin
                new_bbox[:, 1] = bbox[:, 1] * cos - bbox[:, 2] * sin + w * sin
                new_bbox[:, 2] = bbox[:, 2] * cos + bbox[:, 3] * sin
                new_bbox[:, 3] = bbox[:, 3] * cos - bbox[:, 0] * sin + w * sin

            inputs['image'] = image
            inputs['bbox'] = new_bbox

        return inputs


class ToTensor():
    """
    将部分输入内容转为张量
    """

    @type_check(object, dict)
    def __call__(self, inputs):
        """
        Args:
            inputs: 输入数据
        """

        image = inputs['image']

        image = F.to_tensor(image)

        inputs['image'] = image

        return inputs


class ToPILImage():
    """
    将图片张量转换为PIL.IMAGE
    """

    @type_check(object, dict)
    def __call__(self, inputs):
        """
        Args:
            inputs: 输入数据
        """

        image = inputs['image']
        image = F.to_pil_image(image)
        inputs['image'] = image

        return inputs


class Normalize():
    """
    对图片进行标准化处理
    如果先进行rotate、resize等操作，会存在填充背景，是否需要忽略对背景的标准化？
    """

    @type_check(object, EasyDict)
    def __init__(self, cfg):
        """
        Args:
            cfg: 配置文件
        """

        self.mean = cfg.DATASET.MEAN
        self.std = cfg.DATASET.STD

    @type_check(object, dict)
    def __call__(self, inputs):
        """
        Args:
            inputs: 输入数据
        """

        image = inputs['image']

        image = F.normalize(image, self.mean, self.std)

        inputs['image'] = image

        return inputs
