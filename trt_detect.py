# -*- coding: utf-8 -*-
# Author: yohannxu
# Email: yuhannxu@gmail.com
# CreateTime: 2020-09-07 16:14:00
# Description: trt_detect.py

import torch
from torchvision.transforms.functional import to_tensor
import time
import cv2
from model import Model
from yolo.yolo import OnnxPostprocess
from default import cfg
import numpy as np
from PIL import Image
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

trt_logger = trt.Logger()
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

CATEGORIES = [
        '__background__',
        'aeroplane',
        'bicycle',
        'bird',
        'boat',
        'bottle',
        'bus',
        'car',
        'cat',
        'chair',
        'cow',
        'diningtable',
        'dog',
        'horse',
        'motorbike',
        'person',
        'pottedplant',
        'sheep',
        'sofa',
        'train',
        'tvmonitor'
]


def get_engine(engine_path):
    print('Loading engine from {}'.format(engine_path))
    with open(engine_path, 'rb') as f, trt.Runtime(trt_logger) as runtime:
        return runtime.deserialize_cuda_engine(f.read())


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


def allocate_buffers(engine, batch_size):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:

        size = trt.volume(engine.get_binding_shape(binding)) * batch_size
        dims = engine.get_binding_shape(binding)

        # in case batch dimension is -1 (dynamic)
        if dims[0] < 0:
            size *= -1

        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream


def main(engine_path, image_path, image_size):
    post_process = OnnxPostprocess(cfg)
    with get_engine(engine_path) as engine, engine.create_execution_context() as context:
        buffers = allocate_buffers(engine, 1)
        context.set_binding_shape(0, (1, 3, image_size, image_size))

        ori_image = Image.open(image_path)
        w, h = ori_image.size

        for i in range(2):
            t0 = time.time()
            image = ori_image.resize((image_size, image_size))
#            t_image = np.array(image).transpose(1, 2, 0)[None, :] / 255.0
#            t_image = np.ascontiguousarray(t_image.astype(np.float32))
            t_image = to_tensor(image).unsqueeze(0).cpu().numpy()
            print(t_image.dtype, t_image.shape)
            inputs, outputs, bindings, stream = buffers
            inputs[0].host = t_image

            [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
            context.execute_async(bindings=bindings, stream_handle=stream.handle)
            [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
            stream.synchronize()

            outputs = [out.host for out in outputs]

#            for output in outputs:
#                print(output.shape)
#            outputs = [torch.tensor(output.reshape(1, 255, 52 // 2 ** i, 52 // 2 ** i), device=device, dtype=torch.float32) for i, output in enumerate(outputs)]
#            prediction = model.yolo(outputs)[0]

            boxes = outputs[0].reshape(1, 3, -1, 4)
            scores = outputs[1].reshape(1, 3, -1, 80)
            prediction = post_process(boxes, scores)[0]

            print(time.time() - t0)

        box = prediction[:, :4]
        prob = prediction[:, 4]
        label = prediction[:, 5].astype(np.int32) + 1

        box[:, 0::2] *= image_size
        box[:, 1::2] *= image_size
        box[:, 0::2] /= image_size / w
        box[:, 1::2] /= image_size / h
        box = box.astype(np.int32)

        image = np.array(ori_image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        for b, p, l in zip(box, prob, label):
            if p > 0.1:
                print(b)
                cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
                y = b[1] - 10
                if b[1] < 10:
                    y += 30
                cv2.putText(image, '{}_{:.2f}'.format(CATEGORIES[l], p), (b[0], y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imwrite('prediction_trt.jpg', image)
        print('total', time.time() - t0)


if __name__ == '__main__':
    engine_path = 'yolov4_fp16.engine'
    image_path = 'dog.jpg'
    image_size = 416
    main(engine_path, image_path, image_size)
#    get_engine(engine_path)
