# YOLO V3

## paper
[EfficientDet: Scalable and Efficient Object Detection](https://arxiv.org/abs/1911.09070)

[EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)

[Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)

## code reference

https://github.com/xuannianz/EfficientDet

## version

python : 3.7.8

tensorflow : 2.2.0

## train

### hyper parameter

#### STEP 1

epochs = 50

learning_rate = 1e-2

optimizer = Adam

pre trained model from https://github.com/Callidior/keras-applications/releases

#### STEP 2

epochs = 100

learning_rate = cosine decay learning rate start from 1e-2

learning_rate_decay_steps = 30epochs

pre trained model from STEP 1

### our benchmark

#### PASCAL VOC
train with VOC2007 train + VOC2012 trainval

test with VOC2007 val

use RTX-2080Ti FP32 + Tensorflow 2.2

|Model|mAP 50|latency|Params|input size|
|:------:|:---:|:---:|:---:|:---:|
|EfficientDet-b0|49.05%|33|3.9M|512|
|EfficientDet-b1|training| | |640|
|EfficientDet-b2|training| | |768|
|EfficientDet-b3|training| | |896|
|EfficientDet-b4|training| | |1024|
|EfficientDet-b5|training| | |1280|
|EfficientDet-b6|training| | |1280|

### official benchmark

#### COCO

TensorFlow2.1 + CUDA10.1, no TensorRT + Titan-V FP32

|Model|mAP 50|latency|Params|input size|
|:------:|:---:|:---:|:---:|:---:|
|EfficientDet-b0|53.0%|12| |512|
|EfficientDet-b1|59.1%|16| |640|
|EfficientDet-b2|62.7%|23| |768|
|EfficientDet-b3|65.9%|37| |896|
|EfficientDet-b4|68.4%|65| |1024|
|EfficientDet-b5|70.5%|128| |1280|
|EfficientDet-b6|71.5%|169| |1280|

##### EfficientDet-b0
    155 instances of class aeroplane with average precision: 0.7838
    177 instances of class bicycle with average precision: 0.7735
    243 instances of class bird with average precision: 0.7284
    150 instances of class boat with average precision: 0.6393
    252 instances of class bottle with average precision: 0.5341
    114 instances of class bus with average precision: 0.7432
    625 instances of class car with average precision: 0.8182
    190 instances of class cat with average precision: 0.9127
    398 instances of class chair with average precision: 0.4771
    123 instances of class cow with average precision: 0.7931
    112 instances of class diningtable with average precision: 0.5486
    257 instances of class dog with average precision: 0.8554
    180 instances of class horse with average precision: 0.8617
    172 instances of class motorbike with average precision: 0.8173
    2332 instances of class person with average precision: 0.7494
    266 instances of class pottedplant with average precision: 0.3469
    127 instances of class sheep with average precision: 0.7430
    124 instances of class sofa with average precision: 0.6998
    152 instances of class train with average precision: 0.8585
    158 instances of class tvmonitor with average precision: 0.7691
    
    mAP: 0.7227




    
### detect image
![000072](https://user-images.githubusercontent.com/24911666/98916827-fc718500-250e-11eb-8244-dca42f63aaa7.jpg)
![000117](https://user-images.githubusercontent.com/24911666/98916835-fda2b200-250e-11eb-9fea-507cb2d97fa7.jpg)
![000257](https://user-images.githubusercontent.com/24911666/98916837-fe3b4880-250e-11eb-8c11-7aa0322d766d.jpg)
![000305](https://user-images.githubusercontent.com/24911666/98916840-fed3df00-250e-11eb-8704-7b9afde774dc.jpg)
![000661](https://user-images.githubusercontent.com/24911666/98916841-fed3df00-250e-11eb-98bc-b46b606e88d4.jpg)
![000782](https://user-images.githubusercontent.com/24911666/98916845-ff6c7580-250e-11eb-9258-b0e6fdc48d73.jpg)
![000896](https://user-images.githubusercontent.com/24911666/98916846-ff6c7580-250e-11eb-80d8-1ce8744a6fcd.jpg)
![001004](https://user-images.githubusercontent.com/24911666/98916848-00050c00-250f-11eb-8613-92b28481f3b0.jpg)


## TODO
- inference video
- train on EfficientDet 1 ~ 6
- speed up inference
