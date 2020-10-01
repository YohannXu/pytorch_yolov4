# pytorch_yolov4

YOLOv4 implemented by pytorch

## 1 Environment

- python3.6
- torch==1.6.0
- torchvision==0.7.0
- cuda==10.2
- cudnn==8.0.1.13
- tensorrt==7.1.3.4

## 2 Install

### 2.1 下载项目
```bash
git clone https://github.com/YohannXu/pytorch_faster_rcnn.git && cd pytorch_faster_rcnn
```

### 2.2 安装apex
```bash
git clone https://github.com/NVIDIA/apex.git
cd apex && pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
cd ..
```

## 3 Prepare

### 3.1 数据集

只支持coco格式的数据集, 因此需要对其他格式的数据集进行转换, 或自行编写数据加载类。

#### 3.1.1 voc数据集

使用voc2coco.py进行转换, 修改代码中的voc数据集路径后执行
```
python voc2coco.py
```
完成转换。

#### 3.1.2 其他格式数据集

仿照voc2coco.py自行修改。

### 3.2 预训练权重

网络层的结构定义顺序与darknet的cfg略有不同, 因此不能直接加载darknet的权重, 需要先进行转换。

#### 3.2.1 imagenet预训练权重

先下载完整预训练权重 ```csdarknet53-omega_final.weights```: https://drive.google.com/open?id=18jCwaL4SJ-jOvXrZNGHJ5yz44g9zi8Hm

然后使用darknet框架截取前105层的权重。
```
./darknet partial cfg/csdarknet53-omega.cfg csdarknet53-omega_final.weights csdarknet53-omega.conv.105 105
```

最后使用load_backbone.py将权重转换为.pth文件。
```
python load_backbone.py
```

也可直接下载转换好的权重```yolov4_backbone.pth```: https://drive.google.com/file/d/1Dw4nysp0Siripd5UKLUAAtJTr3t6TICF/view?usp=sharing


#### 3.2.2 yolov4权重

先下载yolov4权重 ```yolov4.weights```:

然后使用load.py将权重转换为.pth文件。
```
python load.py
```

也可直接下载转换好的权重```yolov4.pth```:https://drive.google.com/file/d/1Ge6S-abBX_CE-4lwzMjGo1b7KYoCq9j6/view?usp=sharing

### 3.3 配置文件

修改```default.py```数据集路径, 预训练权重路径, 类别数量等参数。

## 4 Usage

### 训练
```
python train.py
```

### 验证
```
python val.py
```

### 推理
```
python infer.py
```

### 推理速度测试
```
python detect.py
```

### 转为onnx格式
```
python pytorch2onnx.py
```

### onnx推理速度测试
```
python onnx_detect.py
```

### 转为tensorrt engine文件
```
trtexec --onnx=yolov4.onnx --explicitBatch --saveEngine=yolov4_fp16.engine --workspace=4096 --fp16
```

### tensorrt加速推理
python trt_detect.py

## 5 训练结果

使用yolov4_backbone.pth进行训练。

|   GPU   |     trainset     |       valset      |  bs  | mini_bs | num_batches | data augment |  mAP   |  AP50  |
| :-----: | :--------------: | :---------------: | :--: | :-----: | :---------: | :----------: | :----: | :----: |
| RTX2070 | voc train2007+12 |    voc val2007    |  64  |    2    |     8000    |      OFF     | 15.37% | 40.19% |
| RTX2070 | voc train2007+12 |    voc val2007    |  64  |    2    |     8000    |      ON      | 17.42% | 43.14% |
| RTX2070 | coco train2014   |    coco val2014   |  64  |    2    |     8000    |      ON      | 8.37%  | 24.02% |
| RTX2070 | coco train2014   |    coco val2014   |  64  |    2    |    40000    |      ON      | %  | % |

使用yolov4.pth进行测试。

|  mAP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |
| :----: | :----: | :----: | :----: | :----: | :----: |
| 46.25% | 72.20% | 50.18% | 26.61% | 51.72% | 59.22% |


## 6 速度测试

|   框架   | 图片尺寸 | 耗时 |
| :------: | :------: | :--: |
| pytorch  |   416    |      |
|   onnx   |   416    |      |
| tensorrt |   416    |      |

## 7 测试结果
