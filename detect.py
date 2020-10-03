# -*- coding: utf-8 -*-
# Author: yohannxu
# Email: yuhannxu@gmail.com
# CreateTime: 2020-09-06 21:03:48
# Description: 速度测试

import time

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor

from categories import COCO_CATEGORIES as CATEGORIES
from default import cfg
from model import Model

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

image_name = 'dog.jpg'
size = 416
weight = 'yolov4.pth'

torch.backends.cudnn.benchmark = True

model = Model(cfg).to(device)
model.load_state_dict(torch.load(weight)['state_dict'])
model.eval()

ori_image = Image.open(image_name).convert('RGB')

w, h = ori_image.size
image = ori_image.resize((size, size))

for i in range(2):
    t0 = time.time()
    t_image = to_tensor(image).unsqueeze(0).to(device)
    t1 = time.time()

    with torch.no_grad():
        x = model.backbone(t_image)
        x = model.spp(*x)
        x = model.pan(*x)
        x = model.head(*x)
        t2 = time.time()

        prediction = model.yolo(x)[0]
    t3 = time.time()

    print('-----------------------------------')
    print('           Preprocess : %f' % (t1 - t0))
    print('      Model Inference : %f' % (t2 - t1))
    print('      Post Processing : %f' % (t3 - t2))
    print('                 total: %f' % (t3 - t0))
    print('-----------------------------------')

box = prediction[:, :4]
prob = prediction[:, 4]
label = prediction[:, 5].astype(np.int32) + 1

box[:, 0::2] *= size
box[:, 1::2] *= size
box[:, 0::2] /= size / w
box[:, 1::2] /= size / h
box = box.astype(np.int32)

image = np.array(ori_image)
image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

for b, p, l in zip(box, prob, label):
    if p > 0.1:
        cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
        y = b[1] - 10
        if b[1] < 10:
            y += 30
        cv2.putText(image, '{}_{:.2f}'.format(CATEGORIES[l], p), (b[0], y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
cv2.imwrite('prediction.jpg', image)
