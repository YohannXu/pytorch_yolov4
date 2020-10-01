# -*- coding: utf-8 -*-
# Author: yohannxu
# Email: yuhannxu@gmail.com
# CreateTime: 2020-08-13 15:59:34
# Description: loss.py


import math
import matplotlib.pyplot as plt
import cv2
import torch.nn.functional as F

import torch
import torch.nn as nn
from ..utils import type_check
from easydict import EasyDict

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


class YOLOLoss(nn.Module):
    def __init__(self, cfg):
        super(YOLOLoss, self).__init__()
        self.base_anchors = torch.tensor(cfg.YOLO.ANCHORS, dtype=torch.float32).to(device)
        self.reductions = cfg.YOLO.REDUCTIONS
        self.masks = cfg.YOLO.MASKS
        self.xy_scales = cfg.YOLO.XY_SCALES
        self.num_classes = cfg.DATASET.NUM_CLASSES
        self.num_anchors = cfg.YOLO.NUM_ANCHORS
        self.iou_thresh = cfg.YOLO.IOU_THRESH
        self.ignore_thresh = cfg.YOLO.IGNORE_THRESH
        self.coord_scale = cfg.YOLO.COORD_SCALE
        self.obj_scale = cfg.YOLO.OBJ_SCALE
        self.noobj_scale = cfg.YOLO.NOOBJ_SCALE
        self.class_scale = cfg.YOLO.CLASS_SCALE
        self.bce = nn.BCELoss(reduction='sum')
        self.smooth_ce = LabelSmoothingLoss(reduction='sum')
        self.ce = nn.CrossEntropyLoss(reduction='sum')
        self.smooth_p = 0.95
        self.smooth_n = 0.05

        # 对每一层anchors除以对应的降采样因子
        anchors = []
        for reduction, mask in zip(self.reductions, self.masks):
            anchor = self.base_anchors[mask] / reduction
            anchors.append(anchor)
        self.anchors = torch.stack(anchors, dim=0).to(device)

    def wh_iou(self, anchors, targets):
        """
        anchors: n x 2
        targets: m x 2
        """
        anchor_area = anchors[:, 0] * anchors[:, 1]  # n
        w = targets[:, 2] - targets[:, 0]  # m
        h = targets[:, 3] - targets[:, 1]  # m
        target_area = w * h  # m
        inter = torch.min(anchors[:, 0][:, None], w[None, :]) * torch.min(anchors[:, 1][:, None], h[None, :])  # n x m
        return inter / (anchor_area[:, None] + target_area[None, :] - inter)  # n x m

    def iou(self, boxes1, boxes2, iou_type):
        """
        boxes1: xywh, pred
        boxes2: xywh, target
        """
        assert iou_type in ['iou', 'giou', 'diou', 'ciou']
        b1x1 = (boxes1[:, 0] - boxes1[:, 2] * 0.5)[:, None]
        b1y1 = (boxes1[:, 1] - boxes1[:, 3] * 0.5)[:, None]
        b1x2 = (boxes1[:, 0] + boxes1[:, 2] * 0.5)[:, None]
        b1y2 = (boxes1[:, 1] + boxes1[:, 3] * 0.5)[:, None]

        b2x1 = (boxes2[:, 0] - boxes2[:, 2] * 0.5)[None, :]
        b2y1 = (boxes2[:, 1] - boxes2[:, 3] * 0.5)[None, :]
        b2x2 = (boxes2[:, 0] + boxes2[:, 2] * 0.5)[None, :]
        b2y2 = (boxes2[:, 1] + boxes2[:, 3] * 0.5)[None, :]

        inter = (torch.min(b1x2, b2x2) - torch.max(b1x1, b2x1)).clamp(0) * (torch.min(b1y2, b2y2) - torch.max(b1y1, b2y1)).clamp(0)
        area1 = boxes1[:, 2] * boxes1[:, 3]
        area2 = boxes2[:, 2] * boxes2[:, 3]
        union = area1[:, None] + area2[None, :] + 1e-16 - inter
        iou = inter / union

        if iou_type == 'iou':
            return iou
        cw = torch.max(b1x2, b2x2) - torch.min(b1x1, b2x1)
        ch = torch.max(b1y2, b2y2) - torch.min(b1y1, b2y1)
        if iou_type == 'giou':
            c_area = cw * ch + 1e-16
            return iou - (c_area - union) / c_area
        c2 = cw ** 2 + ch ** 2 + 1e-16
        rho2 = (boxes1[:, 0] - boxes2[:, 0]) ** 2 + (boxes1[:, 1] - boxes2[:, 1]) ** 2
        if iou_type == 'diou':
            return iou - rho2 / c2
        v = (4 / math.pi ** 2) * (torch.atan(boxes2[:, 2] / boxes2[:, 3]) - torch.atan(boxes1[:, 2] / boxes1[:, 3])) ** 2
        with torch.no_grad():
            alpha = v / (1 - iou + v + 1e-16)
        return iou - (rho2 / c2 + v * alpha)

    def ciou(self, boxes1, boxes2):
        b1x1 = (boxes1[:, 0] - boxes1[:, 2] * 0.5)
        b1y1 = (boxes1[:, 1] - boxes1[:, 3] * 0.5)
        b1x2 = (boxes1[:, 0] + boxes1[:, 2] * 0.5)
        b1y2 = (boxes1[:, 1] + boxes1[:, 3] * 0.5)

        b2x1 = (boxes2[:, 0] - boxes2[:, 2] * 0.5)
        b2y1 = (boxes2[:, 1] - boxes2[:, 3] * 0.5)
        b2x2 = (boxes2[:, 0] + boxes2[:, 2] * 0.5)
        b2y2 = (boxes2[:, 1] + boxes2[:, 3] * 0.5)

        inter = (torch.min(b1x2, b2x2) - torch.max(b1x1, b2x1)).clamp(0) * (torch.min(b1y2, b2y2) - torch.max(b1y1, b2y1)).clamp(0)
        area1 = boxes1[:, 2] * boxes1[:, 3]
        area2 = boxes2[:, 2] * boxes2[:, 3]
        union = area1 + area2 + 1e-16 - inter
        iou = inter / union

        cw = torch.max(b1x2, b2x2) - torch.min(b1x1, b2x1)
        ch = torch.max(b1y2, b2y2) - torch.min(b1y1, b2y1)
        c2 = cw ** 2 + ch ** 2 + 1e-16
        rho2 = (boxes1[:, 0] - boxes2[:, 0]) ** 2 + (boxes1[:, 1] - boxes2[:, 1]) ** 2
        v = (4 / math.pi ** 2) * (torch.atan(boxes2[:, 2] / boxes2[:, 3]) - torch.atan(boxes1[:, 2] / boxes1[:, 3])) ** 2
        with torch.no_grad():
            alpha = v / (1 - iou + v + 1e-16)
        return iou - (rho2 / c2 + v * alpha)

    def forward(self, preds, targets, cats):
        iou_losses = 0.0
        obj_losses = 0.0
        cls_losses = 0.0
        num_targets = 1
        for index, pred in enumerate(preds):
            b, c, h, w = pred.shape
            # b x c x h x w -> b x 3 x (n + 5) x h x w -> b x 3 x h x w x (n + 5)
            pred = pred.view(b, self.num_anchors, self.num_classes + 5, h, w).permute(0, 1, 3, 4, 2)
            anchor = self.anchors[index]

            # 得到预测box
            box_pred = torch.zeros_like(pred[..., :4])
#            box_pred[..., :2] = pred[..., :2].sigmoid() * 2 - 0.5
            box_pred[..., :2] = pred[..., :2].sigmoid() * self.xy_scales[index] - 0.5 * (self.xy_scales[index] - 1)
            box_pred[..., 2:] = pred[..., 2:4].exp() * anchor.view(1, self.num_anchors, 1, 1, 2)
            obj_pred = pred[..., 4].sigmoid()
            cls_pred = pred[..., 5:].sigmoid()

            # 计算anchor与target之间的iou
            target = targets[:, :4] / self.reductions[index]
            target_id = targets[:, 4].long()
            iou_a_t = self.wh_iou(anchor, target)

            # 先得到最大的iou索引
            _, best_anchor_ids = iou_a_t.max(0)

            # 再得到除了最大超过阈值的索引
            # 得到iou大于阈值的anchor target pair, 在yolov3中则是选取iou最大的anchor
            a_id, t_id = torch.nonzero(iou_a_t > self.iou_thresh, as_tuple=False).T

            preserve_ids = []
            for t_index, best_anchor_id in enumerate(best_anchor_ids):
                if best_anchor_id not in a_id[t_id == t_index]:
                    preserve_ids.append(t_index)

            a_id = torch.cat([a_id, best_anchor_ids[preserve_ids]])
            t_id = torch.cat([t_id, torch.arange(best_anchor_ids.shape[0], device=device)[preserve_ids]])

            # 得到target的batch index
            b_id = target_id[t_id]

            t = target[t_id]
            # xyxy -> xywh
            cxcy = (t[:, 2:] + t[:, :2]) / 2
            wh = t[:, 2:] - t[:, :2]
            ij = cxcy.long()
            txy = cxcy - ij
            ti, tj = ij.T
            # 得到gt box坐标
            tbox = torch.cat([txy, wh], dim=1)

            boxes = box_pred[b_id, a_id, tj, ti]

            # 计算iou loss
#            iou = self.iou(boxes, tbox, iou_type='ciou')
            iou = self.ciou(boxes, tbox)
            iou_loss = (1 - iou).sum() * self.coord_scale

            # 计算obj loss
            iou_neg_mask = torch.ones(b, self.num_anchors, h, w)
            # 计算预测boxes和target之间的iou
            xs = torch.arange(w, dtype=torch.float32).to(device)
            ys = torch.arange(h, dtype=torch.float32).to(device)
            ys, xs = torch.meshgrid(xs, ys)
            box_pred[..., 0] += xs
            box_pred[..., 1] += ys
            for i, b in enumerate(box_pred):
                t = target[target_id == i]
                iou = self.iou(b.view(-1, 4), t, iou_type='iou')
                # 当预测boxes和target之间的iou大于阈值时, 忽略其obj loss
                mask = (iou > self.ignore_thresh).sum(1) > 0
                mask = mask.view(b.shape[:-1])
                iou_neg_mask[i, mask] = 0
            iou_neg_mask[b_id, a_id, tj, ti] = 0

            # 上述步骤忽略的预测box如果被划分为正样本, 则仍需计算其obj loss
            # 计算正样本的obj loss
            obj_pos_pred = obj_pred[b_id, a_id, tj, ti]
            obj_pos_t = torch.ones_like(obj_pos_pred)
            obj_pos_loss = self.bce(obj_pos_pred, obj_pos_t)
            # 计算负样本的obj loss
            obj_neg_pred = obj_pred[iou_neg_mask.bool()]
            obj_neg_t = torch.zeros_like(obj_neg_pred)
            obj_neg_loss = self.bce(obj_neg_pred, obj_neg_t)

#            try:
#                print(obj_pos_pred.max(), obj_neg_pred.max())
#            except:
#                pass
#            print(obj_pos_pred)
#            print(obj_pos_loss, obj_neg_loss)
            obj_loss = self.obj_scale * obj_pos_loss + self.noobj_scale * obj_neg_loss

            # 计算cls loss
            cls_pred = cls_pred[b_id, a_id, tj, ti]
#            cls_loss = self.smooth_ce(cls_pred, cats[t_id]) * self.class_scale
            cls_gt = torch.zeros_like(cls_pred)
            cls_gt[torch.arange(cls_pred.shape[0]), cats[t_id] - 1] = 1
            cls_loss = self.bce(cls_pred, cls_gt)
#            cls_loss = self.ce(cls_pred, cats[t_id] - 1)

#            if boxes.shape[0] > 0:
#                iou_loss = iou_loss / boxes.shape[0] / 4
#                cls_loss = cls_loss / boxes.shape[0] / self.num_classes
#            obj_loss = obj_loss / obj_pred.numel()
            iou_losses += iou_loss
            obj_losses += obj_loss
            cls_losses += cls_loss

            num_target = t_id.numel()
        num_targets += num_target

        loss = {
            'iou_loss': iou_losses,
            'obj_loss': obj_losses,
            'cls_loss': cls_losses
        }

        return loss


class YOLOLossLayer(nn.Module):
    def __init__(self, cfg):
        super(YOLOLossLayer, self).__init__()


class LabelSmoothingLoss(nn.Module):
    def __init__(self, reduction='sum'):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = 0.05
        self.conf = 1 - self.smoothing
        self.num_classes = 80
        self.reduction = reduction

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1) - 1, self.conf)
        loss = torch.sum(-true_dist * pred, dim=-1)

        if self.reduction == 'sum':
            return loss.sum()
        if self.reduction == 'mean':
            return loss.mean()


# class YOLOLoss(nn.Module):
#     def __init__(self, cfg):
#         super(YOLOLoss, self).__init__()
#         self.cfg = cfg
#         self.anchors = torch.tensor(cfg.DATASET.ANCHORS, dtype=torch.float32).to(device)
#         self.anchors_masks = cfg.DATASET.ANCHORS_MASKS
#         self.reduction = cfg.DATASET.REDUCTION
#         self.num_classes = cfg.DATASET.NUM_CLASSES
#         self.num_anchors = cfg.DATASET.NUM_ANCHORS
#         self.coord_scale = cfg.YOLO.COORD_SCALE
#         self.obj_scale = cfg.YOLO.OBJ_SCALE
#         self.noobj_scale = cfg.YOLO.NOOBJ_SCALE
#         self.class_scale = cfg.YOLO.CLASS_SCALE
#         self.iou_thresh = cfg.YOLO.IOU_THRESH
# 
#         self.bce = nn.BCELoss(reduction='none').to(device)
#         self.smooth_l1 = nn.SmoothL1Loss(reduction='none').to(device)
#         self.ce = nn.CrossEntropyLoss(reduction='sum').to(device)
# 
#     def iou(self, boxes, targets):
#         """
#         计算预测bbox和真实bbox之间的iou
#         """
#         boxes = boxes.view(-1, 4)  # num_anchors * h * w x 4
#         x_lt = torch.max(boxes[:, 0][:, None], targets[:, 0][None, :])
#         y_lt = torch.max(boxes[:, 1][:, None], targets[:, 1][None, :])
#         x_rb = torch.min(boxes[:, 2][:, None], targets[:, 2][None, :])
#         y_rb = torch.min(boxes[:, 3][:, None], targets[:, 3][None, :])
# 
#         inter = (x_rb - x_lt).clamp(0) * (y_rb - y_lt).clamp(0)
#         area1 = (boxes[:, 2] - boxes[:, 0]).clamp(0) * (boxes[:, 3] - boxes[:, 1]).clamp(0)
#         area2 = (targets[:, 2] - targets[:, 0]) * (targets[:, 3] - targets[:, 1])
#         ious = inter / (area1[:, None] + area2[None, :] - inter)
# 
#         return ious
# 
#     def intersection(self, anchors, targets):
#         """
#         计算anchors与targets之间的iou
#         anchors: Tensor, num_anchors x 2
#         targets: Tensor, n x 4
#         """
#         anchor_area = anchors[:, 0] * anchors[:, 1]
#         w = targets[:, 2] - targets[:, 0]
#         h = targets[:, 3] - targets[:, 1]
#         target_area = w * h
#         inter = torch.min(anchors[:, 0][:, None], w[None, :]) * torch.min(anchors[:, 1][:, None], h[None, :])
#         ious = inter / (anchor_area[:, None] + target_area[None, :] - inter)
# 
#         return ious
# 
#     def forward(self, preds, targets, cats):
#         """
#         Args:
#             preds: list[Tensor], b x num_anchors * (num_classes + 5) x h x w
#             targets: list[Tensor], n x 4, 每张图片的真实边界框坐标
#             cats: list[Tensor], n, 每张图片的真实边界框类别
#         """
#         iou_loss_sum = 0
#         obj_loss_sum = 0
#         cls_loss_sum = 0
#         index = 1
#         num_targets = 1
#         for anchors_masks, pred, reduction in zip(self.anchors_masks, preds, self.reduction):
#             anchors = self.anchors[anchors_masks, :]
#             anchors = anchors / reduction
#             b, c, h, w = pred.shape
#             pred = pred.reshape(b, self.num_anchors, -1, h * w).permute(0, 1, 3, 2)  # b x num_anchors x hw x (num_classes + 5)
#             offset_pred = torch.zeros_like(pred[..., :4])
#             offset_pred[..., :2] = pred[..., :2].sigmoid()
#             offset_pred[..., 2:4] = pred[..., 2:4]
#             iou_pred = pred[..., 4].sigmoid()
#             class_pred = pred[..., 5:]
# 
#             logger.debug('obj pred, level: {}, min: {:.4f}, max: {:.4f}'.format(index, iou_pred.min().item(), iou_pred.max().item()))
#             c_pred = F.softmax(class_pred, dim=-1)
#             logger.debug('cls_pred, level: {}, min: {:.4f}, max: {:.4f}'.format(index, c_pred.min().item(), c_pred.max().item()))
#             index += 1
# 
#             # 得到预测boxes
#             boxes = torch.zeros_like(offset_pred)  # b x num_anchors x hw x 4
#             xs = torch.arange(w, dtype=torch.float32).to(device)
#             ys = torch.arange(h, dtype=torch.float32).to(device)
#             ys, xs = torch.meshgrid(xs, ys)
#             boxes[..., 0] = offset_pred[..., 0].detach() + xs.reshape(-1)
#             boxes[..., 1] = offset_pred[..., 1].detach() + ys.reshape(-1)
#             boxes[..., 2] = offset_pred[..., 2].detach().exp() * anchors[:, 0:1]
#             boxes[..., 3] = offset_pred[..., 3].detach().exp() * anchors[:, 1:2]
#             boxes[..., :2] -= 0.5 * boxes[..., 2:4]
#             boxes[..., 2:4] += boxes[..., :2]
# 
#             offset_mask = torch.zeros(b, self.num_anchors, h * w, 1).to(device)
#             offset_gt = torch.zeros(b, self.num_anchors, h * w, 4).to(device)
#             iou_mask = torch.zeros(b, self.num_anchors, h * w).to(device)
#             iou_neg_mask = torch.ones(b, self.num_anchors, h * w).to(device)
#             iou_gt = torch.zeros(b, self.num_anchors, h * w).to(device)
#             class_mask = torch.zeros(b, self.num_anchors, h * w).to(device)
#             class_gt = torch.zeros(b, self.num_anchors, h * w).to(device)
# 
#             for i in range(b):
#                 target = targets[i] / reduction
#                 iou_box_gt = self.iou(boxes[i], target)
#                 # 若box与任意一个target之间的iou大于阈值
#                 mask = (iou_box_gt > self.iou_thresh).sum(1) > 0
#                 iou_neg_mask[i][mask.view_as(iou_neg_mask[i])] = 0
# 
#                 # 找出与每个target iou最大的anchor
#                 iou_anchor_gt = self.intersection(self.anchors / reduction, target)  # num_anchors x num_target
#                 _, best_anchors = iou_anchor_gt.max(dim=0)  # num_target
# 
#                 target[:, 2:4] = target[:, 2:4] - target[:, :2]
#                 target[:, :2] += 0.5 * target[:, 2:4]
#                 for j in range(target.shape[0]):
#                     if best_anchors[j] in anchors_masks:
#                         cur_n = anchors_masks.index(best_anchors[j])
#                     else:
#                         continue
#                     nx = target[j, 0].floor().int()
#                     ny = target[j, 1].floor().int()
# 
#                     offset_mask[i, cur_n, ny * w + nx, 0] = 2 - target[j, 2] * target[j, 3] / (h * w)
#                     iou_mask[i, cur_n, ny * w + nx] = 1
#                     iou_neg_mask[i, cur_n, ny * w + nx] = 0
#                     class_mask[i, cur_n, ny * w + nx] = 1
# 
#                     offset_gt[i, cur_n, ny * w + nx, 0] = target[j, 0] - nx
#                     offset_gt[i, cur_n, ny * w + nx, 1] = target[j, 1] - ny
#                     offset_gt[i, cur_n, ny * w + nx, 2] = torch.log(target[j, 2] / anchors[cur_n, 0])
#                     offset_gt[i, cur_n, ny * w + nx, 3] = torch.log(target[j, 3] / anchors[cur_n, 1])
#                     iou_gt[i, cur_n, ny * w + nx] = 1
#                     class_gt[i, cur_n, ny * w + nx] = cats[i][j] - 1
# 
#             coord_loss_center = 2.0 * self.coord_scale * (self.bce(offset_pred[..., :2], offset_gt[..., :2]) * offset_mask).sum()
#             coord_loss_wh = 3.0 * self.coord_scale * (self.smooth_l1(offset_pred[..., 2:], offset_gt[..., 2:]) * offset_mask).sum()
#             coord_loss = coord_loss_center + coord_loss_wh
#             iou_pos_loss = self.obj_scale * (self.bce(iou_pred, iou_gt) * iou_mask).sum()
#             iou_neg_loss = self.noobj_scale * (self.bce(iou_pred, iou_gt) * iou_neg_mask).sum()
#             iou_loss = iou_pos_loss + iou_neg_loss
# 
#             class_mask = class_mask > 0
#             cls = class_pred[class_mask]
#             cls_gt = class_gt[class_mask].long()
#             if cls.numel():
#                 class_loss = self.class_scale * self.ce(cls, cls_gt)
#             else:
#                 class_loss = torch.tensor(0.0, device=device)
# 
#             iou_loss_sum += coord_loss
#             obj_loss_sum += iou_loss
#             cls_loss_sum += class_loss
# 
#             num_targets += iou_mask.sum()
# 
#         loss = {
#             'iou_loss': iou_loss_sum / num_targets,
#             'obj_loss': obj_loss_sum / num_targets,
#             'cls_loss': cls_loss_sum / num_targets
#         }
# 
#         return loss