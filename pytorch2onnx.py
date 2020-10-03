# -*- coding: utf-8 -*-
# Author: yohannxu
# Email: yuhannxu@gmail.com
# CreateTime: 2020-09-07 11:36:50
# Description: pytorch2onnx.py

import torch

from default import cfg
from model import OnnxModel

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

model = OnnxModel(cfg).to(device)
model.load_state_dict(torch.load('yolov4.pth')['state_dict'])
model.eval()

x = torch.randn((1, 3, 416, 416), requires_grad=True).to(device)
onnx_filename = 'yolov4.onnx'
input_names = ["input"]
output_names = ['boxes', 'confs']

torch.onnx.export(model,
                  x,
                  onnx_filename,
                  export_params=True,
                  opset_version=11,
                  do_constant_folding=True,
                  input_names=input_names,
                  output_names=output_names,
                  dynamic_axes=None,
                  )
