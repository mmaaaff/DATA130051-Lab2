<h1 align="center">DATA130051 Project1</h1>

<div align="center">周语诠</div>
<div align="center">2024-5-31</div>

## Contents
- [Contents](#contents)
- [Description](#description)
- [Preparation](#preparation)
- [Testing the model](#testing-the-model)

## Description

This lab trains object detection models using the framework MMDetection.

To specify, two models(Faster-RCNN & YOLOv3) are trained seperately on the same dataset VOC2012.

In the root directory there are three config files: 

1. **config1.py** is the Faster-RCNN model with ResNet50 backbone
2. **config2.py** is the Faster_RCNN model with ResNet101 backbone. This is not used in training, so you can ignore it.
3. **config3.py** is the YOLOv3 model with Darknet19 backbone.

In utils, there are tools for data format converting, dataset viewing, etc. Just ignore them.

In results are .events.out files which are for tensorboard visualization of training history and model performance.

***

## Preparation

1. **Environment**
    First make sure you have an MMDetection environment prepared. For installation, see https://mmdetection.readthedocs.io/zh-cn/latest/get_started.html. The versions I use are as follows: \
    mmcv | 2.1.0
    mmdet | 3.3.0
    mmengine | 0.10.4
    It's recommended to install recent versions to avoid compatibility problems. The program will NOT run properly using mmdet 2.x.

2. **Dataset**
   Download converted and splitted data at . Place the folder **converted_data** at root directory.

3. **Model weights**
   Weights of trained model has been uploaded to onedrive. Download **work_dirs** and place it at root directory.

## Testing the model

To test the model on pictures, use following command: 

```bash
python demo/image_demo.py \
    ${IMAGE_FILE} \
    ${CONFIG_FILE} \
    [--weights ${WEIGHTS}] \
    [--device ${GPU_ID}] \
    [--pred-score-thr ${SCORE_THR}]
```

IMAGE_FILE can either be a single image or a directory where testing pictures anre stored. pred-score-thr is optional. Here is an example:

```bash
python mmdetection/demo/image_demo.py ./pictures config3.py --weights work_dirs/config3/epoch_273.pth --out-dir pictures/out/yolo/val
```

