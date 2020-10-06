# -*- coding: utf-8 -*-
# Author: yohannxu
# Email: yuhannxu@gmail.com
# CreateTime: 2020-08-13 15:59:23
# Description: YOLO流程

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict

from ..utils import type_check
from .loss import YOLOLoss

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


class YOLO(nn.Module):

    @type_check(object, EasyDict)
    def __init__(self, cfg):
        """
        Args:
            cfg: 配置文件
        """

        super(YOLO, self).__init__()
        self.anchors = torch.tensor(cfg.YOLO.ANCHORS, dtype=torch.float32).to(device).view(3, 3, 2)
        self.reduction = cfg.YOLO.REDUCTIONS
        self.num_anchors = cfg.YOLO.NUM_ANCHORS
        self.score_thresh = cfg.YOLO.SCORE_THRESH
        self.num_classes = cfg.DATASET.NUM_CLASSES
        self.nms_thresh = cfg.YOLO.NMS_THRESH
        self.loss = YOLOLoss(cfg)
        self.layer1 = YOLOLayer(cfg, self.anchors[0], self.reduction[0])
        self.layer2 = YOLOLayer(cfg, self.anchors[1], self.reduction[1])
        self.layer3 = YOLOLayer(cfg, self.anchors[2], self.reduction[2])

    @type_check(object, np.ndarray, np.ndarray, int)
    def iou(self, area, boxes, index):
        """
        计算某一个box和之后所有boxes之间的iou
        Args:
            area: 提前计算好的所有boxes的面积, 避免重复计算, 假定形状为num x 4
            boxes: 假定形状为num x 4
            index: 索引
        """

        x_tl = np.maximum(boxes[index, 0], boxes[index + 1:, 0])
        y_tl = np.maximum(boxes[index, 1], boxes[index + 1:, 1])
        x_br = np.minimum(boxes[index, 2], boxes[index + 1:, 2])
        y_br = np.minimum(boxes[index, 3], boxes[index + 1:, 3])

        inter = (x_br - x_tl).clip(0) * (y_br - y_tl).clip(0)
        iou = inter / (area[index] + area[index + 1:] - inter)

        return iou

    @type_check(object, np.ndarray, np.ndarray)
    def nms(self, boxes, probs):
        """
        执行NMS, 保留一定数量的boxes
        Args:
            boxes: 预测boxes
            probs: 预测boxes概率
        """
        if boxes.shape[0] == 0:
            return []
        # 根据logits排序
        sorted_inds = probs.argsort()[::-1]
        boxes = boxes[sorted_inds]
        probs = probs[sorted_inds]

        # 计算所有boxes的面积
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        # NMS流程
        keep = []
        # 遍历索引
        num_proposals = boxes.shape[0]
        for i in range(num_proposals):
            # 判断概率是否大于0
            # 如果被抑制, 概率应该为-1, 直接跳过
            # 配合其他nms变种, 可调整此处阈值
            if probs[i] > 0:
                if i != num_proposals - 1:
                    ious = self.iou(area, boxes, i)
                    # 将被抑制的box的概率置为-1
                    # 也可使用softnms将概率降低
                    probs[i + 1:][ious > self.nms_thresh] = -1
                keep.append(i)

        return np.array(sorted_inds[keep])

    @type_check(object, list)
    def inference(self, preds):
        """
        Args:
            preds: head的输出, 网络预测值
        """

        # 得到预测box与score
        boxes_1, scores_1 = self.layer1(preds[0])
        boxes_2, scores_2 = self.layer2(preds[1])
        boxes_3, scores_3 = self.layer3(preds[2])
        # 拼接结果
        boxes = torch.cat([boxes_1, boxes_2, boxes_3], dim=2)
        scores = torch.cat([scores_1, scores_2, scores_3], dim=2)

        # 得到大于概率阈值的mask
        score_threshs = scores > self.score_thresh

        boxes = boxes.unsqueeze(3).repeat_interleave(self.num_classes, 3)

        hw = boxes.shape[2]

        detections_per_img = []
        # 遍历每张图片
        for box, score, score_thresh in zip(boxes, scores, score_threshs):
            # 根据mask筛选出box、score和class_id
            box = box[score_thresh].view(-1, 4)
            score = score[score_thresh].view(-1, 1)
            class_id = torch.arange(self.num_classes).repeat(self.num_anchors, hw, 1)[score_thresh].view(-1, 1).float().to(device)
            if box.shape[0] == 0:
                detections = torch.zeros(0, 6).to(device)
            else:
                detections = torch.cat([box, score, class_id], dim=-1)
            detections_per_img.append(detections)

        detections = []
        for detection in detections_per_img:
            # 在cpu上进行nms会稍微快点
            # 可能是因为都是小矩阵的原因
            detection = detection.cpu().numpy()
            class_ids = detection[:, -1].astype(np.int32)

            nms_detections = []
            # 分类别进行nms
            for i in range(self.num_classes):
                mask = class_ids == i
                box = detection[:, :4][mask]
                prob = detection[:, 4][mask]

                keep = self.nms(box, prob)
                nms_detections.append(detection[mask][keep])

            detection = np.concatenate(nms_detections, axis=0)

            detections.append(detection)

        return detections

    @type_check(object, list, torch.Tensor, torch.Tensor)
    def forward(self, preds, targets=None, cats=None):
        """
        Args:
            preds: head的输出, 网络预测值
            targets: 真实box坐标
            cats: 真实box对应的类别
        """

        if self.training:
            return self.loss(preds, targets, cats)
        return self.inference(preds)


class YOLOLayer(nn.Module):
    @type_check(object, EasyDict, torch.Tensor, int)
    def __init__(self, cfg, anchors, reduction):
        super(YOLOLayer, self).__init__()
        self.anchors = anchors / reduction
        self.num_anchors = cfg.YOLO.NUM_ANCHORS

    @type_check(object, torch.Tensor)
    def forward(self, pred):
        b, c, h, w = pred.shape
        pred = pred.reshape(b, self.num_anchors, -1, h * w).permute(0, 1, 3, 2)
        # 这里是否需要乘上xy_scale?
        pred[..., :2] = pred[..., :2].sigmoid()
        boxes = pred[..., :4]
        conf_preds = pred[..., 4].sigmoid()
#        class_preds = F.softmax(pred[..., 5:], dim=-1)
        class_preds = pred[..., 5:].sigmoid()

        xs = torch.arange(w, dtype=torch.float32).to(device)
        ys = torch.arange(h, dtype=torch.float32).to(device)
        ys, xs = torch.meshgrid(xs, ys)
        boxes[..., 0] = boxes[..., 0] + xs.reshape(-1)
        boxes[..., 1] = boxes[..., 1] + ys.reshape(-1)
        boxes[..., 2] = boxes[..., 2].exp() * self.anchors[:, 0:1]
        boxes[..., 3] = boxes[..., 3].exp() * self.anchors[:, 1:2]
        boxes[..., 0::2] = boxes[..., 0::2] / w
        boxes[..., 1::2] = boxes[..., 1::2] / h
        boxes[..., :2] -= 0.5 * boxes[..., 2:4]
        boxes[..., 2:4] += boxes[..., :2]

        scores = conf_preds.unsqueeze(-1) * class_preds

        return boxes, scores


class OnnxYOLO(nn.Module):
    """
    和YOLO类差不多, 修改了一些onnx不支持的写法
    """

    def __init__(self, cfg):
        super(OnnxYOLO, self).__init__()
        self.anchors = np.array(cfg.YOLO.ANCHORS, dtype=np.float32).reshape(3, 3, 2)
        self.reductions = cfg.YOLO.REDUCTIONS
        self.num_anchors = cfg.YOLO.NUM_ANCHORS
        self.layer1 = OnnxYOLOLayer(cfg, self.anchors[0], self.reductions[0])
        self.layer2 = OnnxYOLOLayer(cfg, self.anchors[1], self.reductions[1])
        self.layer3 = OnnxYOLOLayer(cfg, self.anchors[2], self.reductions[2])

    def forward(self, preds):
        boxes_1, scores_1 = self.layer1(preds[0])
        boxes_2, scores_2 = self.layer2(preds[1])
        boxes_3, scores_3 = self.layer3(preds[2])
        boxes = torch.cat([boxes_1, boxes_2, boxes_3], dim=2)
        scores = torch.cat([scores_1, scores_2, scores_3], dim=2)
        return boxes, scores


class OnnxYOLOLayer(nn.Module):

    def __init__(self, cfg, anchors, reduction):
        super(OnnxYOLOLayer, self).__init__()
        self.anchors = anchors / reduction
        self.num_anchors = cfg.DATASET.NUM_ANCHORS

    def forward(self, output):
        pred = output.reshape(output.size(0), self.num_anchors, -1, output.size(2) * output.size(3)).permute(0, 1, 3, 2)
        boxes = torch.cat([torch.sigmoid(pred[..., :2]), pred[..., 2:4]], dim=3)
        conf_pred = torch.sigmoid(pred[..., 4])
        class_pred = F.softmax(pred[..., 5:], dim=-1)

        grid_x = np.expand_dims(np.expand_dims(np.expand_dims(np.linspace(0, output.size(3) - 1, output.size(3)),
                                                              axis=0).repeat(output.size(2), 0), axis=0), axis=0).repeat(self.num_anchors, 1)
        grid_y = np.expand_dims(np.expand_dims(np.expand_dims(np.linspace(0, output.size(2) - 1, output.size(2)),
                                                              axis=1).repeat(output.size(3), 1), axis=0), axis=0).repeat(self.num_anchors, 1)
        grid_x = grid_x.reshape(pred.size(0), pred.size(1), pred.size(2))
        grid_y = grid_y.reshape(pred.size(0), pred.size(1), pred.size(2))

        anchor_w = np.expand_dims(self.anchors[:, 0:1], 0).repeat(output.size(2) * output.size(3), 2)
        anchor_h = np.expand_dims(self.anchors[:, 1:2], 0).repeat(output.size(2) * output.size(3), 2)

        xs = boxes[..., 0] + torch.tensor(grid_x, device=device, dtype=torch.float32)
        ys = boxes[..., 1] + torch.tensor(grid_y, device=device, dtype=torch.float32)

        ws = torch.exp(boxes[..., 2]) * torch.tensor(anchor_w, device=device, dtype=torch.float32)
        hs = torch.exp(boxes[..., 3]) * torch.tensor(anchor_h, device=device, dtype=torch.float32)

        xs = xs / output.size(3)
        ys = ys / output.size(2)
        ws = ws / output.size(3)
        hs = hs / output.size(2)

        x1 = xs - 0.5 * ws
        y1 = ys - 0.5 * hs
        x2 = x1 + ws
        y2 = y1 + hs
        boxes = torch.stack([x1, y1, x2, y2], dim=3)

        scores = conf_pred.unsqueeze(-1) * class_pred

        return boxes, scores


class OnnxPostprocess(nn.Module):
    def __init__(self, cfg):
        super(OnnxPostprocess, self).__init__()
        self.num_anchors = cfg.YOLO.NUM_ANCHORS
        self.num_classes = cfg.DATASET.NUM_CLASSES
        self.score_thresh = cfg.YOLO.SCORE_THRESH
        self.nms_thresh = cfg.YOLO.NMS_THRESH

    def iou(self, area, boxes, index):
        """
        计算某一个box和之后所有boxes之间的iou
        Args:
            area: 提前计算好的所有boxes的面积, 避免重复计算, 假定形状为num x 4
            boxes: 假定形状为num x 4
            index: 索引
        """
        x_tl = np.maximum(boxes[index, 0], boxes[index + 1:, 0])
        y_tl = np.maximum(boxes[index, 1], boxes[index + 1:, 1])
        x_br = np.minimum(boxes[index, 2], boxes[index + 1:, 2])
        y_br = np.minimum(boxes[index, 3], boxes[index + 1:, 3])

        inter = (x_br - x_tl).clip(0) * (y_br - y_tl).clip(0)
        iou = inter / (area[index] + area[index + 1:] - inter)

        return iou

    def nms(self, boxes, probs):
        """
        执行NMS, 保留一定数量的boxes
        Args:
            boxes: 预测boxes
            probs: 预测boxes概率
        """
        if boxes.shape[0] == 0:
            return []
        # 根据logits排序
        sorted_inds = probs.argsort()[::-1]
        boxes = boxes[sorted_inds]
        probs = probs[sorted_inds]

        # 计算所有boxes的面积
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        # NMS流程
        keep = []
        # 遍历索引
        num_proposals = boxes.shape[0]
        for i in range(num_proposals):
            if probs[i] > 0:
                if i != num_proposals - 1:
                    ious = self.iou(area, boxes, i)
                    probs[i + 1:][ious > self.nms_thresh] = -1
                keep.append(i)

        return np.array(sorted_inds[keep])

    def forward(self, boxes, scores):
        score_threshs = scores > self.score_thresh
        boxes = np.expand_dims(boxes, 3).repeat(self.num_classes, 3)
        hw = boxes.shape[2]

        detections_per_level = []
        detections_per_img = []
        for box, score, score_thresh in zip(boxes, scores, score_threshs):
            box = box[score_thresh].reshape(-1, 4)
            score = score[score_thresh].reshape(-1, 1)
            class_id = np.arange(self.num_classes)[None, :].repeat(hw, 0)[None, :].repeat(
                self.num_anchors, 0)[score_thresh].reshape(-1, 1).astype(np.float32)
            if box.shape[0] == 0:
                detections = np.zeros((0, 6))
            else:
                detections = np.concatenate([box, score, class_id], axis=-1)
            detections_per_img.append(detections)

        detections_per_level.append(detections_per_img)

        detections_per_img = list(zip(*detections_per_level))

        detections = []
        for detection in detections_per_img:
            detection = np.concatenate(detection, axis=0)
            class_ids = detection[:, -1].astype(np.int32)

            nms_detections = []
            for i in range(self.num_classes):
                mask = class_ids == i
                box = detection[:, :4][mask]
                prob = detection[:, 4][mask]

                keep = self.nms(box, prob)
                nms_detections.append(detection[mask][keep])

            detection = np.concatenate(nms_detections, axis=0)

            detections.append(detection)

        return detections
