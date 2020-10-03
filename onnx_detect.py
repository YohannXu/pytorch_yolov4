# -*- coding: utf-8 -*-
# Author: yohannxu
# Email: yuhannxu@gmail.com
# CreateTime: 2020-09-07 12:10:24
# Description: onnx_detect.py

import time

import cv2
import numpy as np
import onnxruntime
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor

from categories import COCO_CATEGORIES as CATEGORIES
from default import cfg
from yolo.yolo import OnnxPostprocess

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

post_process = OnnxPostprocess(cfg)

image_name = 'dog.jpg'
ori_image = Image.open(image_name)
w, h = ori_image.size

session = onnxruntime.InferenceSession('yolov4.onnx')
resize_h = session.get_inputs()[0].shape[2]
resize_w = session.get_inputs()[0].shape[3]
image = ori_image.resize((resize_w, resize_h))

for i in range(2):
    t0 = time.time()
    image_t = to_tensor(image).unsqueeze(0).cpu().numpy()
    t1 = time.time()

    input_name = session.get_inputs()[0].name

    features = session.run(None, {input_name: image_t})
    t2 = time.time()

    boxes = features[0]
    scores = features[1]
    prediction = post_process(boxes, scores)[0]
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

box[:, 0::2] *= resize_w
box[:, 1::2] *= resize_h
box[:, 0::2] /= resize_w / w
box[:, 1::2] /= resize_h / h
box = box.astype(np.int32)

image = np.array(ori_image)
image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

for b, p, l in zip(box, prob, label):
    if p > 0.3:
        cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
        y = b[1] - 10
        if b[1] < 10:
            y += 30
        cv2.putText(image, '{}_{:.2f}'.format(CATEGORIES[l], p), (b[0], y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
cv2.imwrite('prediction_onnx.jpg', image)
print('total', time.time() - t0)
