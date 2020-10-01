# -*- coding: utf-8 -*-
# Author: yohannxu
# Email: yuhannxu@gmail.com
# CreateTime: 2020-08-11 03:50:21
# Description: default.py

import os
import numpy as np
import pandas as pd
import cv2
from glob import glob
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

from easydict import EasyDict as edict

cfg = edict()

cfg.OUTPUT = 'coco2_saved_models'

cfg.DATASET = edict()
cfg.DATASET.NUM_CLASSES = 80
cfg.DATASET.FREQUENCY = 5000
cfg.DATASET.USE_MOSAIC = True
cfg.DATASET.MOSAIC_RESIZE_RANGE = [0.8, 0.9]
cfg.DATASET.MEAN = (0.485, 0.456, 0.406)
cfg.DATASET.STD = (0.229, 0.224, 0.225)
cfg.DATASET.SIZE = 608
cfg.DATASET.SIZES = [608]
cfg.DATASET.TEST_SIZE = 608
cfg.DATASET.FLIP_HORIZONTAL_PROB = 0.5
cfg.DATASET.BRIGHTNESS = 0.1
cfg.DATASET.CONTRAST = 0.1
cfg.DATASET.SATURATION = 0.1
cfg.DATASET.HUE = 0.1
cfg.DATASET.ERASER_SIZE = [0.1, 0.3]
cfg.DATASET.ROTATE_RANGE = [-5, 5]
cfg.DATASET.GRID_RANGE = [0.1, 0.3]
cfg.DATASET.GRID_RATIOS = [0.4, 0.6]
cfg.DATASET.GRID_PROBS = [0.3, 0.7]
# cfg.DATASET.TRAIN_ROOT = '/datasets/voc/train2007+12'
# cfg.DATASET.TRAIN_ANNO = '/datasets/voc/annotations/instances_train2007+12.json'
# cfg.DATASET.VAL_ROOT = '/datasets/voc/val2007'
# cfg.DATASET.VAL_ANNO = '/datasets/voc/annotations/instances_val2007.json'
cfg.DATASET.TRAIN_ROOT = '/datasets/coco/train2014'
cfg.DATASET.TRAIN_ANNO = '/datasets/coco/annotations/instances_train2014.json'
cfg.DATASET.VAL_ROOT = '/datasets/coco/val2014'
cfg.DATASET.VAL_ANNO = '/datasets/coco/annotations/instances_val2014.json'
cfg.DATASET.ANCHORS = [(10, 13), (16, 30), (33, 23), (30, 61), (62, 45), (59, 119), (116, 90), (156, 198), (373, 326)]
cfg.DATASET.ANCHORS_MASKS = [(0, 1, 2), (3, 4, 5), (6, 7, 8)]
cfg.DATASET.REDUCTION = [8, 16, 32]
cfg.DATASET.NUM_ANCHORS = 3
cfg.DATASET.MINI_BATCH_SIZE = 2
cfg.DATASET.BATCH_SIZE = 64
cfg.DATASET.VAL_BATCH_SIZE = 1
cfg.DATASET.NUM_WORKERS = 4

cfg.TRAIN = edict()
cfg.TRAIN.NUM_BATCHES = 40000
cfg.TRAIN.MIX_LEVEL = 'O0'
cfg.TRAIN.LOGDIR = 'tensorboard'
cfg.TRAIN.LOG_INTERVAL = 20
cfg.TRAIN.SAVE_INTERVAL = 2000

cfg.OPTIMIZER = edict()
cfg.OPTIMIZER.BASE_LR = 0.0013
cfg.OPTIMIZER.WEIGHT_DECAY = 0.0005
cfg.OPTIMIZER.MOMENTUM = 0.949
cfg.OPTIMIZER.STEPS = (28000, 36000)
cfg.OPTIMIZER.GAMMA = 0.1
cfg.OPTIMIZER.COSINE_GAMMA = 0.99
cfg.OPTIMIZER.WARMUP_FACTOR = 1.0 / 20
cfg.OPTIMIZER.WARMUP_ITERS = 1000
cfg.OPTIMIZER.WARMUP_METHOD = 'power'
cfg.OPTIMIZER.WARMUP_POWER = 4

cfg.BACKBONE = edict()
cfg.BACKBONE.ACTIVATION_FN = 'mish'

cfg.SPP = edict()
cfg.SPP.ACTIVATION_FN = 'leaky'

cfg.PAN = edict()
cfg.PAN.ACTIVATION_FN = 'leaky'

cfg.HEAD = edict()
cfg.HEAD.ACTIVATION_FN = 'leaky'

cfg.YOLO = edict()
cfg.YOLO.ANCHORS = [[12, 16], [19, 36], [40, 28], [36, 75], [76, 55], [72, 146], [142, 110], [192, 243], [459, 401]]
cfg.YOLO.MASKS = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
cfg.YOLO.REDUCTIONS = [8, 16, 32]
cfg.YOLO.NUM_ANCHORS = 3
cfg.YOLO.IOU_THRESH = 0.213
cfg.YOLO.IGNORE_THRESH = 0.7
cfg.YOLO.SCORE_THRESH = 0.005
cfg.YOLO.NMS_THRESH = 0.45

cfg.YOLO.XY_SCALES = [1.2, 1.1, 1.05]
cfg.YOLO.COORD_SCALE = 0.07
cfg.YOLO.OBJ_SCALE = 1.0
cfg.YOLO.NOOBJ_SCALE = 1.0
cfg.YOLO.CLASS_SCALE = 1.0
