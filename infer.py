# -*- coding: utf-8 -*-
# Author: yohannxu
# Email: yuhannxu@gmail.com
# CreateTime: 2020-08-19 22:15:10
# Description: infer.py

import os
import sys

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from categories import COCO_CATEGORIES, VOC_CATEGORIES
from default import cfg
from model import Model
from yolo.data import Collater, InferenceDataset, build_transforms
from yolo.utils import last_checkpoint

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


def inference():
    is_train_or_val = False
    save_dir = 'results'

    dataset = InferenceDataset(
        image_dir='infer_images',
        transforms=build_transforms(cfg, is_train=False)
    )
    collater = Collater(cfg, is_train_or_val)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        collate_fn=collater,
        shuffle=False,
        num_workers=2
    )
    model = Model(cfg).to(device)

    checkpoint = last_checkpoint(cfg)
    if 'voc' in checkpoint:
        CATEGORIES = VOC_CATEGORIES
    elif 'coco' in checkpoint:
        CATEGORIES = COCO_CATEGORIES
    else:
        print('dataset not support!')
        sys.exit()

    if checkpoint:
        print('loading {}'.format(checkpoint))
        checkpoint = torch.load(checkpoint)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    else:
        print('weight not found')
        sys.exit()

    model.eval()
    for data in dataloader:
        ori_images = data['ori_images']
        images = data['images'].to(device)
        sizes = data['sizes']
        ratios = data['ratios']
        names = data['names']
        offsets = data['offsets']

        with torch.no_grad():
            detections = model(images)

        for ori_image, detection, size, ratio, name, offset in zip(ori_images, detections, sizes, ratios, names, offsets):
            box = detection[:, :4]
            prob = detection[:, 4]
            label = detection[:, 5].astype(np.int32) + 1

            box[:, 0::2] *= size[0].item()
            box[:, 1::2] *= size[1].item()
            box[:, 0::2] -= offset[0]
            box[:, 1::2] -= offset[1]
            box[:, 0::2] /= ratio[0]
            box[:, 1::2] /= ratio[1]
            box = box.astype(np.int32)

            ori_image = np.array(ori_image)
            ori_image = cv2.cvtColor(ori_image, cv2.COLOR_RGB2BGR)

            for b, p, l in zip(box, prob, label):
                if p > 0.3:
                    cv2.rectangle(ori_image, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
                    y = b[1] - 10
                    if b[1] < 10:
                        y += 30
                    cv2.putText(ori_image, '{}_{:.2f}'.format(CATEGORIES[l], p), (b[0], y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            cv2.imwrite('{}/{}'.format(save_dir, os.path.basename(name)), ori_image)


if __name__ == '__main__':
    import time
    start = time.time()
    inference()
    end = time.time()
    print(end - start)
