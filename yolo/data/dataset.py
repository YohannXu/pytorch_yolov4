# -*- coding: utf-8 -*-
# Author: yohannxu
# Email: yuhannxu@gmail.com
# CreateTime: 2020-08-17 17:02:33
# Description: dataset.py

import json
import os
import random
from glob import glob
import numpy as np
import math

import cv2
import torch
from easydict import EasyDict
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import BatchSampler, Dataset
from torchvision.transforms import transforms as T

from ..utils import type_check


class COCODataset(Dataset):
    """
    COCO格式数据集
    用于训练及验证
    """

    @type_check(object, EasyDict, T.Compose, bool)
    def __init__(self, cfg, transforms=None, is_train=True):
        """
        Args:
            cfg: str, 配置文件
            transforms: 图片预处理
            is_train: bool, 训练还是验证
        """
        super(COCODataset, self).__init__()
        self.is_train = is_train
        if is_train:
            self.root = cfg.DATASET.TRAIN_ROOT
            self.anno_file = cfg.DATASET.TRAIN_ANNO
        else:
            self.root = cfg.DATASET.VAL_ROOT
            self.anno_file = cfg.DATASET.VAL_ANNO
        self.use_mosaic = cfg.DATASET.USE_MOSAIC

        self.transforms = transforms
        if 'coco' in self.root:
            # COCO数据集中类别索引不是从1~80,因此手动调整到1~80
            with open('yolo/data/classes.json') as f:
                self.classes = json.load(f)
        else:
            self.classes = {str(i): i for i in range(1, 21)}

        # 加载数据集
        self.coco = COCO(self.anno_file)
        # 得到所有图片索引
        ids = list(sorted(self.coco.imgs.keys()))
        # 将不包含bbox标记的图片去掉
        self.ids = []
        for img_id in ids:
            anno_ids = self.coco.getAnnIds(img_id)
            if anno_ids:
                annos = self.coco.loadAnns(anno_ids)
                # 如果所有annos的宽高都大于1,就保留该图片
                # 只要有一个小于等于1,就丢弃该图片
                # TODO 可以只丢弃该anno
                if not all(any(scale <= 1 for scale in anno['bbox'][2:]) for anno in annos):
                    self.ids.append(img_id)

    def __len__(self):
        return len(self.ids)

    @type_check(object, int)
    def __getitem__(self, idx):
        if self.use_mosaic and self.is_train:
            img_id = self.ids[idx]
            img_ids = [img_id] + [self.ids[i] for i in random.sample(range(len(self)), 3)]
            image_names = [self.coco.loadImgs(ids=img_id)[0]['file_name'] for img_id in img_ids]
            images = [cv2.cvtColor(cv2.imread(os.path.join(self.root, image_name)), cv2.COLOR_BGR2RGB) for image_name in image_names]
            targets = [self.coco.imgToAnns[img_id] for img_id in img_ids]
            bboxes = []
            cats = []
            for target in targets:
                bbox = []
                cat = []
                for ann in target:
                    bbox.append(ann['bbox'])
                    cat.append(self.classes[str(ann['category_id'])])
                bbox = torch.tensor(bbox)
                bbox[:, 2:] = bbox[:, :2] + bbox[:, 2:]
                cat = torch.tensor(cat)
                bboxes.append(bbox)
                cats.append(cat)
            data = {
                    'image': images,
                    'bbox': bboxes,
                    'cat': cats,
                    'name': image_names[0],
                    'img_id': img_ids[0]
                    }
        else:
            img_id = self.ids[idx]
            image_name = self.coco.loadImgs(ids=img_id)[0]['file_name']
            image = Image.open(os.path.join(self.root, image_name)).convert('RGB')

            target = self.coco.imgToAnns[img_id]
            # 提取出annotation中与目标检测相关的部分
            bbox = []
            cat = []
            for ann in target:
                bbox.append(ann['bbox'])
                cat.append(self.classes[str(ann['category_id'])])
            bbox = torch.tensor(bbox)
            bbox[:, 2:] += 1
            bbox[:, 2:] = bbox[:, :2] + bbox[:, 2:]
            cat = torch.tensor(cat)

            data = {
                'image': image,
                'bbox': bbox,
                'cat': cat,
                'name': image_name,
                'img_id': img_id
            }

        if self.transforms:
            data = self.transforms(data)

        return data

    def get_info(self, index):
        img_id = self.ids[index]
        info = self.coco.imgs[img_id]
        return info


class InferenceDataset(Dataset):
    """
    推理时的数据集类
    """

    @type_check(object, str, T.Compose)
    def __init__(self, image_dir, transforms=None):
        """
        Args:
            image_dir: str, 需要推理的图片文件夹路径
            transforms: 图片预处理
        """
        super(InferenceDataset, self).__init__()
        self.image_names = glob('{}/*'.format(image_dir))
        self.transforms = transforms

    def __len__(self):
        return len(self.image_names)

    @type_check(object, int)
    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image = Image.open(image_name)

        data = {
            'ori_image': image,
            'image': image,
            'name': image_name
        }

        if self.transforms:
            data = self.transforms(data)

        return data


class DataSampler(BatchSampler):
    """
    加载数据时的Sampler
    """

    @type_check(object, Dataset, EasyDict, int, bool)
    def __init__(self, dataset, cfg, start_iter=0, is_train=True):
        """
        Args:
            dataset: 数据集
            cfg: 配置文件
            start_iter: 当前迭代次数
            is_train: 训练还是验证
        """

        self.dataset = dataset
        self.start_iter = start_iter
        self.is_train = is_train
        if self.is_train:
            self.batch_size = cfg.DATASET.MINI_BATCH_SIZE
            self.num_iters = cfg.TRAIN.NUM_BATCHES * cfg.DATASET.BATCH_SIZE / cfg.DATASET.MINI_BATCH_SIZE
        else:
            self.batch_size = cfg.DATASET.VAL_BATCH_SIZE
            self.num_iters = math.ceil(len(dataset) / self.batch_size)

    def prepare_batches(self):
        """
        生成batch index
        """

        splits = self.sample_ids.split(self.batch_size)
        batches = [s.tolist() for s in splits]

        return batches

    def __iter__(self):
        iteration = self.start_iter
        while iteration <= self.num_iters:
            # 训练状态, 打乱图片顺序
            # 每过一个epoch, 重新打乱顺序
            if self.is_train:
                self.sample_ids = torch.randperm(len(self.dataset))
            else:
                self.sample_ids = torch.arange(len(self.dataset))
            batches = self.prepare_batches()
            for batch in batches:
                yield batch
                iteration += 1
                if iteration > self.num_iters:
                    break

    def __len__(self):
        return self.num_iters


class Collater():
    """
    用于拼接一个batch中的数据
    """

    @type_check(object, EasyDict, bool)
    def __init__(self, cfg, is_train_or_val=True):
        self.cfg = cfg
        self.is_train_or_val = is_train_or_val

    @type_check(object, list)
    def __call__(self, batch):
        if self.is_train_or_val:
            origin_images = [item['image'] for item in batch]
            bboxes = [item['bbox'] for item in batch]
            cats = [item['cat'] for item in batch]
            ratios = [item['ratio'] for item in batch]
            names = [item['name'] for item in batch]
            img_ids = [item['img_id'] for item in batch]
            offsets = [item['offset'] for item in batch]
            inds = torch.cat(
                    [
                        torch.full((len(bbox), 1), i, dtype=torch.long) for i, bbox in enumerate(bboxes)
                        ]
                    )
            bboxes = torch.cat(bboxes, dim=0)
            bboxes = torch.cat([bboxes, inds], dim=-1)
            cats = torch.cat(cats)
        else:
            ori_images = [item['ori_image'] for item in batch]
            origin_images = [item['image'] for item in batch]
            ratios = [item['ratio'] for item in batch]
            names = [item['name'] for item in batch]
            offsets = [item['offset'] for item in batch]

        # 拼接缩放后图片
        images = torch.stack(origin_images, dim=0)

        # 拼接图片尺寸
        sizes = torch.zeros(len(origin_images), 2, dtype=torch.float32)
        for i, image in enumerate(origin_images):
            sizes[i, 0] = image.size(1)
            sizes[i, 1] = image.size(2)

        if self.is_train_or_val:
            data = {
                'images': images,
                'bboxes': bboxes,
                'cats': cats,
                'sizes': sizes,
                'ratios': ratios,
                'names': names,
                'img_ids': img_ids,
                'offsets': offsets
            }
        else:
            data = {
                'ori_images': ori_images,
                'images': images,
                'sizes': sizes,
                'ratios': ratios,
                'names': names,
                'offsets': offsets
            }

        return data
