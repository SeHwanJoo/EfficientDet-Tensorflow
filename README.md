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

|Model|mAP 50|latency| |
|:------:|:---:|:---:|:---:|
|EfficientDet-b0|49.05%| | |
|EfficientDet-b1|training| | |
|EfficientDet-b2|training| | |
|EfficientDet-b3|training| | |
|EfficientDet-b4|training| | |
|EfficientDet-b5|training| | |
|EfficientDet-b6|training| | |

### official benchmark

#### COCO

|Model|mAP 50|latency| |
|:------:|:---:|:---:|:---:|
|EfficientDet-b0|49.05%| | |
|EfficientDet-b1|training| | |
|EfficientDet-b2|training| | |
|EfficientDet-b3|training| | |
|EfficientDet-b4|training| | |
|EfficientDet-b5|training| | |
|EfficientDet-b6|training| | |

##### EfficientDet-b0
    60.55% = aeroplane AP
    59.61% = bicycle AP
    34.77% = bird AP
    29.81% = boat AP
    21.59% = bottle AP
    54.67% = bus AP
    68.62% = car AP
    53.91% = cat AP
    30.35% = chair AP
    48.36% = cow AP
    42.37% = diningtable AP
    46.38% = dog AP
    65.49% = horse AP
    63.44% = motorbike AP
    68.24% = person AP
    27.39% = pottedplant AP
    45.50% = sheep AP
    45.06% = sofa AP
    63.74% = train AP
    51.23% = tvmonitor AP
    
    mAP = 49.05%
    
### detect image
![000001](https://user-images.githubusercontent.com/24911666/95420197-0c2d0500-0976-11eb-8af2-15c815635cae.jpg)
![000070](https://user-images.githubusercontent.com/24911666/95420201-0df6c880-0976-11eb-8ecf-d8f64e4dc973.jpg)
![000108](https://user-images.githubusercontent.com/24911666/95420206-10592280-0976-11eb-88a2-53c33626155f.jpg)
![000185](https://user-images.githubusercontent.com/24911666/95420214-12bb7c80-0976-11eb-8068-a73495bb9f14.jpg)
![000216](https://user-images.githubusercontent.com/24911666/95420223-14854000-0976-11eb-815b-554e08151fce.jpg)
![000239](https://user-images.githubusercontent.com/24911666/95420225-164f0380-0976-11eb-9e89-8f22dec9aa4d.jpg)
![000280](https://user-images.githubusercontent.com/24911666/95420247-1c44e480-0976-11eb-904e-a8086e4497b4.jpg)


## TODO
- inference video
- train on EfficientDet 1 ~ 6
- speed up inference
