# -*- coding: utf-8 -*-
# Author: yohannxu
# Email: yuhannxu@gmail.com
# CreateTime: 2020-08-11 03:50:21
# Description: default.py


from easydict import EasyDict as edict

cfg = edict()

# 训练时权重保存目录
cfg.OUTPUT = 'coco2_saved_models'

cfg.DATASET = edict()
# 类别数量
cfg.DATASET.NUM_CLASSES = 80
# cfg.DATASET.TRAIN_ROOT = '/datasets/voc/train2007+12'
# cfg.DATASET.TRAIN_ANNO = '/datasets/voc/annotations/instances_train2007+12.json'
# cfg.DATASET.VAL_ROOT = '/datasets/voc/val2007'
# cfg.DATASET.VAL_ANNO = '/datasets/voc/annotations/instances_val2007.json'
# 训练集图片路径
cfg.DATASET.TRAIN_ROOT = '/datasets/coco/train2014'
# 训练集annotations路径
cfg.DATASET.TRAIN_ANNO = '/datasets/coco/annotations/instances_train2014.json'
# 验证集图片路径
cfg.DATASET.VAL_ROOT = '/datasets/coco/val2014'
# 验证集annotations路径
cfg.DATASET.VAL_ANNO = '/datasets/coco/annotations/instances_val2014.json'

# 训练时网络输入图片大小
cfg.DATASET.SIZE = 608
# 测试时网络输入图片大小
cfg.DATASET.TEST_SIZE = 608
# 多尺度训练的尺度列表
cfg.DATASET.SIZES = [608]
# 多尺度训练时的变化频率
cfg.DATASET.FREQUENCY = 5000
# 是否使用mosaic数据增强
cfg.DATASET.USE_MOSAIC = True
# mosaic数据增强中图片的缩放范围
cfg.DATASET.MOSAIC_RESIZE_RANGE = [0.8, 0.9]
# mosaic数据增强中的背景填充色
cfg.DATASET.MOSAIC_COLOR = (0, 0, 0)
# 亮度调整
cfg.DATASET.BRIGHTNESS = 0.1
# 对比度调整
cfg.DATASET.CONTRAST = 0.1
# 饱和度调整
cfg.DATASET.SATURATION = 0.1
# 色度调整
cfg.DATASET.HUE = 0.1
# 随机擦除概率
cfg.DATASET.ERASER_PROB = 0.5
# 随机擦除范围大小
cfg.DATASET.ERASER_SIZE_RATIO = [0.1, 0.3]
# 随机擦除填充色
cfg.DATASET.ERASER_COLOR = (0, 0, 0,)
# 旋转概率
cfg.DATASET.ROTATE_PROB = 0.5
# 旋转角度范围
cfg.DATASET.ROTATE_RANGE = [-5, 5]
# 水平翻转概率
cfg.DATASET.FLIP_HORIZONTAL_PROB = 0.5
cfg.DATASET.GRID_RANGE = [0.1, 0.3]
cfg.DATASET.GRID_RATIOS = [0.4, 0.6]
cfg.DATASET.GRID_PROBS = [0.3, 0.7]
# 缩放时背景填充色
cfg.DATASET.RESIZE_COLOR = (0, 0, 0)
# normalize均值
cfg.DATASET.MEAN = (0.485, 0.456, 0.406)
# normalize方差
cfg.DATASET.STD = (0.229, 0.224, 0.225)

cfg.DATASET.MINI_BATCH_SIZE = 2
cfg.DATASET.BATCH_SIZE = 64
cfg.DATASET.VAL_BATCH_SIZE = 1
cfg.DATASET.NUM_WORKERS = 4

cfg.TRAIN = edict()
# 训练的batch数量
cfg.TRAIN.NUM_BATCHES = 40000
# 混合精度
cfg.TRAIN.MIX_LEVEL = 'O0'
# tensorboard保存路径
cfg.TRAIN.LOGDIR = 'tensorboard'
# 日志输出及保存间隔(mini batch)
cfg.TRAIN.LOG_INTERVAL = 20
# 权重的保存间隔(mini batch)
cfg.TRAIN.SAVE_INTERVAL = 2000
# 预训练权重
cfg.TRAIN.PRETRAIN_WEIGHT = 'yolov4_backbone.pth'

cfg.OPTIMIZER = edict()
cfg.OPTIMIZER.BASE_LR = 0.0013
cfg.OPTIMIZER.WEIGHT_DECAY = 0.0005
cfg.OPTIMIZER.MOMENTUM = 0.949
# 学习率下降间隔
cfg.OPTIMIZER.STEPS = (28000, 36000)
# 学习率下降比率
cfg.OPTIMIZER.GAMMA = 0.1
# cosine模拟退火学习率下降参数
cfg.OPTIMIZER.COSINE_GAMMA = 0.99
# warmup初始比率
cfg.OPTIMIZER.WARMUP_FACTOR = 1.0 / 20
# warmup对应batch数量
cfg.OPTIMIZER.WARMUP_ITERS = 1000
# warmup方法类型
cfg.OPTIMIZER.WARMUP_METHOD = 'power'
# 指数型warmup的参数
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
# anchors
cfg.YOLO.ANCHORS = [[12, 16], [19, 36], [40, 28], [36, 75], [76, 55], [72, 146], [142, 110], [192, 243], [459, 401]]
# 每一层对应的anchor mask
cfg.YOLO.MASKS = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
# 每一层的降采样系数
cfg.YOLO.REDUCTIONS = [8, 16, 32]
# 每一层对应的anchor数量
cfg.YOLO.NUM_ANCHORS = 3
# 判断box是否为正样本的阈值
cfg.YOLO.IOU_THRESH = 0.213
# 判断是否忽略box的阈值
cfg.YOLO.IGNORE_THRESH = 0.7
# 测试时的概率阈值
cfg.YOLO.SCORE_THRESH = 0.005
# nms阈值
cfg.YOLO.NMS_THRESH = 0.45
# 扩大xy坐标预测的范围
cfg.YOLO.XY_SCALES = [1.2, 1.1, 1.05]
# loss权重
cfg.YOLO.COORD_SCALE = 0.07
cfg.YOLO.OBJ_SCALE = 1.0
cfg.YOLO.NOOBJ_SCALE = 1.0
cfg.YOLO.CLASS_SCALE = 1.0
