# -*- coding: utf-8 -*-
# Author: yohannxu
# Email: yuhannxu@gmail.com
# CreateTime: 2020-08-17 17:02:38
# Description: transforms.py

import torch
import random
import math
import cv2
import numpy as np

import torchvision.transforms.functional as F
from easydict import EasyDict
from PIL import Image
from torchvision import transforms as T

from ..utils import type_check


@type_check(EasyDict, bool)
def build_transforms(cfg, is_train=True):
    """
    数据预处理
    """
    mean = cfg.DATASET.MEAN
    std = cfg.DATASET.STD

    if is_train:
        size = cfg.DATASET.SIZE
        flip_horizontal_prob = cfg.DATASET.FLIP_HORIZONTAL_PROB
        brightness = cfg.DATASET.BRIGHTNESS
        contrast = cfg.DATASET.CONTRAST
        saturation = cfg.DATASET.SATURATION
        hue = cfg.DATASET.HUE
        eraser_size = cfg.DATASET.ERASER_SIZE
        rotate_range = cfg.DATASET.ROTATE_RANGE

        transforms = T.Compose(
            [
                MultiSizes(cfg),
                Mosaic(cfg),
                ColorJitter(
                    brightness,
                    contrast,
                    saturation,
                    hue
                ),
                RandomEraser(eraser_size),
                Rotate(rotate_range),
                RandomHorizontalFlip(flip_horizontal_prob),
                Resize(size),
                ToTensor(),
            ]
        )
    else:
        size = cfg.DATASET.TEST_SIZE
        transforms = T.Compose(
            [
                Resize(size),
                ToTensor(),
            ]
        )

    return transforms


class MultiSizes():
    def __init__(self, cfg):
        self.cfg = cfg
        self.sizes = cfg.DATASET.SIZES
        self.frequency = cfg.DATASET.FREQUENCY
        self.time_to_change = self.frequency

    def __call__(self, inputs):
        self.time_to_change -= 1
        if self.time_to_change == 0:
            self.cfg.DATASET.SIZE = random.choice(self.sizes)
            self.time_to_change = self.frequency
        return inputs


class Mosaic():
    def __init__(self, cfg, interpolation=Image.BILINEAR):
        self.resize_range = cfg.DATASET.MOSAIC_RESIZE_RANGE
        self.size = cfg.DATASET.SIZE

    def __call__(self, inputs):
        images = inputs['image']
        bboxes = inputs['bbox']
        cats = inputs['cat']

        if type(images) is not list:
            return inputs

        new_image = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        new_bboxes = []
        new_cats = []
        cx, cy = [random.randint(self.size // 4, self.size // 4 * 3) for _ in range(2)]
        resize_ratio = random.uniform(self.resize_range[0], self.resize_range[1])
        size = int(self.size * resize_ratio)

        for index, (image, bbox, cat) in enumerate(zip(images, bboxes, cats)):
            w, h = image.shape[:2]

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

            bbox[:, 0::2] *= ratio_w
            bbox[:, 1::2] *= ratio_h

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

            offset_w = x1a - x1b
            offset_h = y1a - y1b
            bbox[:, 0::2] += offset_w
            bbox[:, 1::2] += offset_h

            out_mask = (bbox[:, [0, 2]] <= 0).all(1) | (bbox[:, [0, 2]] >= self.size).all(1) | (bbox[:, [1, 3]] <= 0).all(1) | (bbox[:, [1, 3]] >= self.size).all(1)
            mask = ~out_mask
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

    @type_check(object, int, int)
    def __init__(self, size, interpolation=Image.BILINEAR):
        """
        Args:
            cfg, 配置文件
            sizes: 多尺度
            interpolation: 插值算法
        """
        self.size = size
        self.interpolation = interpolation

    @type_check(object, dict)
    def __call__(self, inputs):
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

        new_image = Image.new('RGB', (self.size, self.size))
        new_image.paste(image, (offset_w, offset_h))

        inputs['image'] = new_image
        inputs['ratio'] = [ratio_w, ratio_h]
        inputs['offset'] = [offset_w, offset_h]

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

    @type_check(object, float)
    def __init__(self, prob=0.5):
        self.prob = prob

    @type_check(object, dict)
    def __call__(self, inputs):
        image = inputs['image']
        bbox = inputs['bbox']

        if random.random() < self.prob:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            w, h = image.size
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

    @type_check(object, float, float, float, float)
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        """
        Args:
            brightness:, float, 亮度
            contrast: float, 对比度
            saturation: float, 饱和度
            hue: float, 色调
        """
        self.color_jitter = T.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue
        )

    @type_check(object, dict)
    def __call__(self, inputs):
        image = inputs['image']
        image = self.color_jitter(image)
        inputs['image'] = image

        return inputs


class RandomEraser():
    """
    对每一个box进行随机擦除
    如果擦除区域太大，可能会遮盖目标的重要部分
    如果擦除区域太小，又不会起什么作用
    """
    @type_check(object, list, float, tuple)
    def __init__(self, size_ratio, prob=0.5, color=(0, 0, 0)):
        self.prob = prob
        self.size_ratio = size_ratio
        self.color = color

    @type_check(object, dict)
    def __call__(self, inputs):
        image = inputs['image']
        bbox = inputs['bbox']

        for box in bbox:
            if random.random() > self.prob:
                w = box[2] - box[0]
                h = box[3] - box[1]
                size_w = random.randint(int(w * self.size_ratio[0]), int(w * self.size_ratio[1]))
                size_h = random.randint(int(h * self.size_ratio[0]), int(h * self.size_ratio[1]))
                range_w = random.randint(int(box[0]), int(box[2] - size_w))
                range_h = random.randint(int(box[1]), int(box[3] - size_h))
                rec = Image.new('RGB', (size_w, size_h))
                image.paste(rec, (range_w, range_h))

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
    @type_check(object, list, float)
    def __init__(self, angle, prob=0.5):
        self.prob = prob
        self.angle = angle

    @type_check(object, dict)
    def __call__(self, inputs):
        if random.random() > self.prob:
            image = inputs['image']
            w, h = image.size
            bbox = inputs['bbox']

            angle = random.randint(self.angle[0], self.angle[1])
            image = image.rotate(angle, expand=True)
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
        image = inputs['image']
        image = F.to_pil_image(image)
        inputs['image'] = image

        return inputs


class Normalize():
    """
    对图片进行标准化处理
    如果先进行rotate、resize等操作，会存在填充背景，是否需要忽略对背景的标准化？
    """

    @type_check(object, list, list)
    def __init__(self, mean, std):
        """
        Args:
            mean: float, 均值
            std: float, 标准差
        """
        self.mean = mean
        self.std = std

    @type_check(object, dict)
    def __call__(self, inputs):
        image = inputs['image']

        image = F.normalize(image, self.mean, self.std)

        inputs['image'] = image

        return inputs
