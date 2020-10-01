# pytorch_yolov4

YOLOv4 implemented by pytorch

## 1 Environment

- python3.6
- torch==1.6.0
- torchvision==0.7.0
- cuda==10.2
- cudnn==8.0.1.13
- tensorrt==7.1.3.4

其他版本未验证

## 2 Install

```bash
# 1 下载项目
git clone https://github.com/YohannXu/pytorch_faster_rcnn.git && cd pytorch_faster_rcnn

# 2 安装apex(可选)
git clone https://github.com/NVIDIA/apex.git
cd apex && pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
cd ..
```

## 3 Prepare

# 3.1 数据集
```bash
只支持coco格式的数据集, 因此需要进行转换，或自己编写数据加载类。

# 1 voc2coco

使用voc2coco.py进行转换, 修改voc数据集路径后执行
python voc2coco.py

# 2 其他格式数据集

仿照voc2coco.py自行修改。
```

# 3.2 预训练权重

网络层的定义顺序与darknet的cfg略有不同, 因此需要对darknet的权重进行转换才能正确使用。

```bash
# 1 imagenet预训练权重

先下载完整预训练权重 `csdarknet53-omega_final.weights`: https://drive.google.com/open?id=18jCwaL4SJ-jOvXrZNGHJ5yz44g9zi8Hm

然后使用命令 `./darknet partial cfg/csdarknet53-omega.cfg csdarknet53-omega_final.weights csdarknet53-omega.conv.105 105`

最后使用load_backbone.py进行顺序的转换。
`python load_backbone.py`

也可直接下载转换好的权重

# 2 yolov4权重

先下载yolov4权重 `yolov4.weights`:

然后使用load.py进行转换。
`python load.py`
```

# 3.3 配置文件

```
必须修改数据加载路径, 预训练权重路径, 类别数量
其他可选
```

## 4 Usage

```bash
# 训练
python train.py

# 验证
python val.py

# 推理
python infer.py

# 推理速度测试
python detect.py

# 转为onnx格式
python pytorch2onnx.py

# onnx推理速度测试
python onnx_detect.py

# 转为tensorrt engine文件
trtexec 

# tensorrt加速推理
python trt_detect.py
```

## 5 训练结果

|   GPU   | backbone | 初始化权重 | 迭代次数 | batch size |  mAP   |
| :-----: | :-----:  | :--------: | :------: | :--------: | :----: |
| RTX2070 | resnet50 |   caffe2   |  90000   |     2      | 26.48% |
| RTX2070 | resnet50 |  pytorch   |  90000   |     2      | 12.80% |

## 6 速度测试

## 7 测试结果
