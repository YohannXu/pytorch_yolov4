# -*- coding: utf-8 -*-
# Author: yohannxu
# Email: yuhannxu@gmail.com
# CreateTime: 2020-08-19 04:17:06
# Description: train.py

import cv2
import matplotlib.pyplot as plt
import datetime
import logging
import os
import time
import shutil
import sys
import torch
import torch.optim as optim
from apex import amp
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from default import cfg
from model import Model
from yolo.data import COCODataset, Collater, DataSampler, build_transforms
from yolo.utils import Metric, WarmupMultiStepLR, last_checkpoint, WarmupCosineLR

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


def train():
    is_train = True
    model = Model(cfg, pretrained=True).to(device)
    batch_size = cfg.DATASET.BATCH_SIZE
    mini_batch_size = cfg.DATASET.MINI_BATCH_SIZE

    lr = cfg.OPTIMIZER.BASE_LR / batch_size
    weight_decay = cfg.OPTIMIZER.WEIGHT_DECAY * batch_size
    optimizer = optim.SGD(model.parameters(), lr, momentum=cfg.OPTIMIZER.MOMENTUM, weight_decay=weight_decay)
#    model, optimizer = amp.initialize(model, optimizer, opt_level=cfg.TRAIN.MIX_LEVEL)

    checkpoint = last_checkpoint(cfg)
    if 'model_final.pth' in checkpoint:
        print('training has completed!')
        sys.exit()
    if checkpoint:
        checkpoint = torch.load(checkpoint)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        metric = checkpoint['metric']
        step = checkpoint['step']
        global_step = checkpoint['global_step']
    else:
        metric = Metric()
        step = 1
        global_step = 1

    scheduler = WarmupMultiStepLR(
        optimizer,
        cfg,
        step - 2
    )

#    scheduler = WarmupCosineLR(
#        optimizer,
#        cfg.TRAIN.NUM_BATCHES,
#        cfg.OPTIMIZER.WARMUP_FACTOR,
#        cfg.OPTIMIZER.WARMUP_ITERS,
#        cfg.OPTIMIZER.WARMUP_METHOD,
#        cfg.OPTIMIZER.COSINE_GAMMA,
#        step - 2
#    )

    # 加载训练数据集
    dataset = COCODataset(
        cfg,
        transforms=build_transforms(cfg),
        is_train=is_train
    )
    collater = Collater(cfg)
    sampler = DataSampler(dataset, cfg, global_step, is_train)
    dataloader = DataLoader(
        dataset,
        collate_fn=collater,
        batch_sampler=sampler,
        num_workers=cfg.DATASET.NUM_WORKERS
    )

    if os.path.exists(cfg.TRAIN.LOGDIR):
        shutil.rmtree(cfg.TRAIN.LOGDIR)
    writer = SummaryWriter(log_dir=cfg.TRAIN.LOGDIR)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    stream_handler = logging.StreamHandler()
    logger.addHandler(stream_handler)
    file_handler = logging.FileHandler('train.log')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    model.train()

    backward_step = int(batch_size / mini_batch_size)
    max_iters = cfg.TRAIN.NUM_BATCHES * backward_step

    start = time.time()
    for data in dataloader:
        images = data['images'].to(device)
        bboxes = data['bboxes'].to(device)
        cats = data['cats'].to(device)
#        bboxes = [bbox.to(device) for bbox in data['bboxes']]
#        cats = [cat.to(device) for cat in data['cats']]

        losses = model(images, bboxes, cats)

        loss = losses['iou_loss'] + losses['obj_loss'] + losses['cls_loss']

        if torch.isnan(loss):
            print('loss has became nan, skip this step')
            global_step += 1
            continue

        loss.backward()
#        with amp.scale_loss(loss, optimizer) as scaled_loss:
#            scaled_loss.backward()

        if global_step % backward_step == 0:

#            max_grad, min_grad = 0, 0
#            for p in model.parameters():
#                max_g = p.grad.max()
#                min_g = p.grad.min()
#                if max_g > max_grad:
#                    max_grad = max_g
#                if min_g < min_grad:
#                    min_grad = min_g
#            print('max_grad:', max_grad.item(), 'min_grad:', min_grad.item())

            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            step += 1

        batch_time = time.time() - start
        start = time.time()

#        losses['iou_loss'] /= mini_batch_size
#        losses['obj_loss'] /= mini_batch_size
#        losses['cls_loss'] /= mini_batch_size
#        loss /= mini_batch_size

        metric.update('iou_loss', losses['iou_loss'])
        metric.update('obj_loss', losses['obj_loss'])
        metric.update('cls_loss', losses['cls_loss'])
        metric.update('loss', loss)
        metric.update('time', batch_time)

        writer.add_scalar('Loss/loss', loss, global_step)
        writer.add_scalar('Loss/iou_loss', losses['iou_loss'], global_step)
        writer.add_scalar('Loss/obj_loss', losses['obj_loss'], global_step)
        writer.add_scalar('Loss/cls_loss', losses['cls_loss'], global_step)

        eta_time = (max_iters - global_step) * metric['time'].global_avg
        eta_time = str(datetime.timedelta(seconds=int(eta_time)))
        # 打印日志
        if global_step % cfg.TRAIN.LOG_INTERVAL == 0:
            message = metric.delimiter.join(
                [
                    'step: {global_step} / {max_iters}',
                    'eta: {eta}',
                    '{metric}',
                    'lr: {lr:.8f}',
                    'max mem: {memory:.0f}'
                ]
            ).format(
                global_step=global_step,
                max_iters=max_iters,
                eta=eta_time,
                metric=str(metric),
                lr=optimizer.param_groups[0]['lr'],
                memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
            )
            logger.info(message)

        # 定时保存模型
        if global_step % cfg.TRAIN.SAVE_INTERVAL == 0:
            checkpoint = {
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'metric': metric,
                'step': step,
                'global_step': global_step
            }

            if not os.path.exists(cfg.OUTPUT):
                os.makedirs(cfg.OUTPUT)
            torch.save(checkpoint, '{}/model_{:04d}.pth'.format(
                cfg.OUTPUT, global_step))

        global_step += 1

        if global_step == max_iters:
            torch.save({'state_dict': model.state_dict()}, '{}/model_final.pth'.format(cfg.OUTPUT))


if __name__ == '__main__':
    train()
