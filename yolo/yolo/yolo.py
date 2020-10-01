# -*- coding: utf-8 -*-
# Author: yohannxu
# Email: yuhannxu@gmail.com
# CreateTime: 2020-08-13 15:59:23
# Description: yolo.py

import os
import numpy as np
import pandas as pd
import cv2
from glob import glob
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import nms
from .loss import YOLOLoss
from easydict import EasyDict
from ..utils import type_check

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


class YOLO(nn.Module):
    def __init__(self, cfg):
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

    def nms_cpu(self, boxes, confs, nms_thresh=0.5, min_mode=False):
        if boxes.shape[0] == 0:
            return []
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1) * (y2 - y1)
        order = confs.argsort()[::-1]

        keep = []
        while order.size > 0:
            idx_self = order[0]
            idx_other = order[1:]

            keep.append(idx_self)

            xx1 = np.maximum(x1[idx_self], x1[idx_other])
            yy1 = np.maximum(y1[idx_self], y1[idx_other])
            xx2 = np.minimum(x2[idx_self], x2[idx_other])
            yy2 = np.minimum(y2[idx_self], y2[idx_other])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h

            if min_mode:
                over = inter / np.minimum(areas[order[0]], areas[order[1:]])
            else:
                over = inter / (areas[order[0]] + areas[order[1:]] - inter)

            inds = np.where(over <= nms_thresh)[0]
            order = order[inds + 1]

        return np.array(keep)

#    def process(self, preds):
#        boxes_list, scores_list = [], []
#        for anchors, output, reduction in zip(self.anchors, preds, self.reduction):
#            anchors = anchors / reduction
#            pred = output.reshape(output.size(0), self.num_anchors, -1, output.size(2) * output.size(3)).permute(0, 1, 3, 2)  # b x num_anchors x hw x (num_classes + 5)
##            pred[..., :2] = torch.sigmoid(pred[..., :2])
##            boxes = pred[..., :4]
#            boxes = torch.cat([torch.sigmoid(pred[..., :2]), pred[..., 2:4]], dim=3)
#            iou_preds = torch.sigmoid(pred[..., 4])
#            class_preds = F.softmax(pred[..., 5:], dim=-1)
##            class_preds = pred[..., 5:].sigmoid()
#
#            # 得到预测boxes
##            xs = torch.arange(w, dtype=torch.float32).to(device)
##            ys = torch.arange(h, dtype=torch.float32).to(device)
##            ys, xs = torch.meshgrid(xs, ys)
##
##            ys = ys.reshape(-1).unsqueeze(0).unsqueeze(0).repeat(1, self.num_anchors, 1)
##            xs = xs.reshape(-1).unsqueeze(0).unsqueeze(0).repeat(1, self.num_anchors, 1)
#
#            grid_x = np.expand_dims(np.expand_dims(np.expand_dims(np.linspace(0, output.size(3) - 1, output.size(3)), axis=0).repeat(output.size(2), 0), axis=0), axis=0).repeat(self.num_anchors, 1)
#            grid_y = np.expand_dims(np.expand_dims(np.expand_dims(np.linspace(0, output.size(2) - 1, output.size(2)), axis=1).repeat(output.size(3), 1), axis=0), axis=0).repeat(self.num_anchors, 1)
#            grid_x = grid_x.reshape(pred.size(0), pred.size(1), pred.size(2))
#            grid_y = grid_y.reshape(pred.size(0), pred.size(1), pred.size(2))
#
#            anchor_w = np.expand_dims(anchors[:, 0:1], 0).repeat(output.size(2) * output.size(3), 2)
#            anchor_h = np.expand_dims(anchors[:, 1:2], 0).repeat(output.size(2) * output.size(3), 2)
#
##            boxes[..., 0] = boxes[..., 0] + xs.reshape(-1)
##            boxes[..., 1] = boxes[..., 1] + ys.reshape(-1)
##            boxes[..., 2] = boxes[..., 2].exp() * anchors[:, 0:1]
##            boxes[..., 3] = boxes[..., 3].exp() * anchors[:, 1:2]
#
##            x1 = boxes[..., 0] + xs
##            y1 = boxes[..., 1] + ys
##            x2 = torch.exp(boxes[:, :, :, 2]) * torch.tensor(anchors[:, 0:1], dtype=torch.float32, device=device)
##            y2 = torch.exp(boxes[:, :, :, 3]) * torch.tensor(anchors[:, 1:2], dtype=torch.float32, device=device)
#            xs = boxes[..., 0] + torch.tensor(grid_x, device=device, dtype=torch.float32)
#            ys = boxes[..., 1] + torch.tensor(grid_y, device=device, dtype=torch.float32)
#
#            ws = torch.exp(boxes[..., 2]) * torch.tensor(anchor_w, device=device, dtype=torch.float32)
#            hs = torch.exp(boxes[..., 3]) * torch.tensor(anchor_h, device=device, dtype=torch.float32)
##            bboxes = torch.stack([x1, y1, x2, y2], dim=3)
#
##            boxes[..., 0::2] = boxes[..., 0::2] / w
##            boxes[..., 1::2] = boxes[..., 1::2] / h
##            boxes[..., :2] -= 0.5 * boxes[..., 2:4]
##            boxes[..., 2:4] += boxes[..., :2]
#
##            ax1 = bboxes[..., 0] / w
##            ay1 = bboxes[..., 1] / h
##            ax2 = bboxes[..., 2] / w
##            ay2 = bboxes[..., 3] / h
##            bbboxes = torch.stack([ax1, ay1, ax2, ay2], dim=3)
#            xs = xs / output.size(3)
#            ys = ys / output.size(2)
#            ws = ws / output.size(3)
#            hs = hs / output.size(2)
#
##            bx1 = bbboxes[..., 0] - 0.5 * bbboxes[..., 2]
##            by1 = bbboxes[..., 1] - 0.5 * bbboxes[..., 3]
##            bx2 = bx1 + bbboxes[..., 2]
##            by2 = by1 + bbboxes[..., 3]
##            bbbboxes = torch.stack([bx1, by1, bx2, by2], dim=3)
#            x1 = xs - 0.5 * ws
#            y1 = ys - 0.5 * hs
#            x2 = x1 + ws
#            y2 = y1 + hs
#            boxes = torch.stack([x1, y1, x2, y2], dim=3)
#
#            scores = iou_preds.unsqueeze(-1) * class_preds
#
#            boxes_list.append(boxes)
#            scores_list.append(scores)
##        return boxes_list, scores_list
#
#        concat_boxes = torch.cat(boxes_list, dim=2)
#        concat_scores = torch.cat(scores_list, dim=2)
#
#        return concat_boxes, concat_scores

    def post_process(self, boxes, scores):
        b, c, h, w = boxes.shape
#        boxes = boxes.reshape(b, self.num_anchors, c // self.num_anchors, 4)
#        scores = scores.reshape(b, self.num_anchors, c // self.num_anchors, 80)

        detections_per_level = []
#        if torch.is_tensor(boxes):
#            boxes = boxes.cpu().numpy()
#        if torch.is_tensor(scores):
#            scores = scores.cpu().numpy()
        score_threshs = scores > self.score_thresh

        if torch.is_tensor(boxes):
            boxes = boxes.unsqueeze(3).repeat_interleave(self.num_classes, 3)
        else:
            boxes = np.expand_dims(boxes, 3).repeat(self.num_classes, 3)

        hw = boxes.shape[2]

        detections_per_img = []
        # 遍历每张图片
        for box, score, score_thresh in zip(boxes, scores, score_threshs):
            if torch.is_tensor(boxes):
                box = box[score_thresh].view(-1, 4)
                score = score[score_thresh].view(-1, 1)
                class_id = torch.arange(self.num_classes).repeat(self.num_anchors, hw, 1)[score_thresh].view(-1, 1).float().to(device)
            else:
                box = box[score_thresh].reshape(-1, 4)
                score = score[score_thresh].reshape(-1, 1)
                class_id = np.arange(self.num_classes)[None, :].repeat(hw, 0)[None, :].repeat(self.num_anchors, 0)[score_thresh].reshape(-1, 1).astype(np.float32)
            if box.shape[0] == 0:
                if torch.is_tensor(boxes):
                    detections = torch.zeros(0, 6).to(device)
                else:
                    detections = np.zeros((0, 6))
            else:
                if torch.is_tensor(boxes):
                    detections = torch.cat([box, score, class_id], dim=-1)
                else:
                    detections = np.concatenate([box, score, class_id], axis=-1)
            detections_per_img.append(detections)

        detections_per_level.append(detections_per_img)

        detections_per_img = list(zip(*detections_per_level))

        detections = []
        for detection in detections_per_img:
            if torch.is_tensor(boxes):
                detection = torch.cat(detection, dim=0).cpu().numpy()
            else:
                detection = np.concatenate(detection, axis=0)
            class_ids = detection[:, -1].astype(np.int32)

            nms_detections = []
            for i in range(self.num_classes):
                mask = class_ids == i
                box = detection[:, :4][mask]
                prob = detection[:, 4][mask]

                keep = self.nms_cpu(box, prob, self.nms_thresh)
                nms_detections.append(detection[mask][keep])

            detection = np.concatenate(nms_detections, axis=0)

            detections.append(detection)

        return detections

    def inference(self, preds):
        boxes_1, scores_1 = self.layer1(preds[0])
        boxes_2, scores_2 = self.layer2(preds[1])
        boxes_3, scores_3 = self.layer3(preds[2])
        boxes = torch.cat([boxes_1, boxes_2, boxes_3], dim=2)
        scores = torch.cat([scores_1, scores_2, scores_3], dim=2)

        score_threshs = scores > self.score_thresh

        boxes = boxes.unsqueeze(3).repeat_interleave(self.num_classes, 3)

        hw = boxes.shape[2]

        detections_per_img = []
        # 遍历每张图片
        for box, score, score_thresh in zip(boxes, scores, score_threshs):
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
            detection = detection.cpu().numpy()
            class_ids = detection[:, -1].astype(np.int32)

            nms_detections = []
            for i in range(self.num_classes):
                mask = class_ids == i
                box = detection[:, :4][mask]
                prob = detection[:, 4][mask]

                keep = self.nms_cpu(box, prob, self.nms_thresh)
                nms_detections.append(detection[mask][keep])

            detection = np.concatenate(nms_detections, axis=0)

            detections.append(detection)

        return detections

    def forward(self, preds, targets=None, cats=None):
        if self.training:
            return self.loss(preds, targets, cats)
        return self.inference(preds)


class YOLOLayer(nn.Module):
    def __init__(self, cfg, anchors, reduction):
        super(YOLOLayer, self).__init__()
        self.anchors = anchors / reduction
        self.num_anchors = cfg.DATASET.NUM_ANCHORS

    def forward(self, pred):
        b, c, h, w = pred.shape
        pred = pred.reshape(b, self.num_anchors, -1, h * w).permute(0, 1, 3, 2)
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
#        print('max score:', scores.max())

        return boxes, scores


class OnnxYOLO(nn.Module):
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

        grid_x = np.expand_dims(np.expand_dims(np.expand_dims(np.linspace(0, output.size(3) - 1, output.size(3)), axis=0).repeat(output.size(2), 0), axis=0), axis=0).repeat(self.num_anchors, 1)
        grid_y = np.expand_dims(np.expand_dims(np.expand_dims(np.linspace(0, output.size(2) - 1, output.size(2)), axis=1).repeat(output.size(3), 1), axis=0), axis=0).repeat(self.num_anchors, 1)
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

    def nms_cpu(self, boxes, confs, nms_thresh=0.5, min_mode=False):
        if boxes.shape[0] == 0:
            return []
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1) * (y2 - y1)
        order = confs.argsort()[::-1]

        keep = []
        while order.size > 0:
            idx_self = order[0]
            idx_other = order[1:]

            keep.append(idx_self)

            xx1 = np.maximum(x1[idx_self], x1[idx_other])
            yy1 = np.maximum(y1[idx_self], y1[idx_other])
            xx2 = np.minimum(x2[idx_self], x2[idx_other])
            yy2 = np.minimum(y2[idx_self], y2[idx_other])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h

            if min_mode:
                over = inter / np.minimum(areas[order[0]], areas[order[1:]])
            else:
                over = inter / (areas[order[0]] + areas[order[1:]] - inter)

            inds = np.where(over <= nms_thresh)[0]
            order = order[inds + 1]

        return np.array(keep)

    def forward(self, boxes, scores):
        score_threshs = scores > self.score_thresh
        boxes = np.expand_dims(boxes, 3).repeat(self.num_classes, 3)
        hw = boxes.shape[2]

        detections_per_level = []
        detections_per_img = []
        for box, score, score_thresh in zip(boxes, scores, score_threshs):
            box = box[score_thresh].reshape(-1, 4)
            score = score[score_thresh].reshape(-1, 1)
            class_id = np.arange(self.num_classes)[None, :].repeat(hw, 0)[None, :].repeat(self.num_anchors, 0)[score_thresh].reshape(-1, 1).astype(np.float32)
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

                keep = self.nms_cpu(box, prob, self.nms_thresh)
                nms_detections.append(detection[mask][keep])

            detection = np.concatenate(nms_detections, axis=0)

            detections.append(detection)

        return detections
