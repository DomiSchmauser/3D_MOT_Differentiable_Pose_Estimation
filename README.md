
# 3D Multi-Object Tracking with Differentiable Pose Estimation

## Introduction
This a Pytorch implementation of our work "3D Multi-Object Tracking with Differentiable Pose Estimation".
In this project, we present a novel framework for 3D Multi-Object tracking in indoor scenes. To train and evaluate our framework, 
on the our dataset MOTFront, please refer to the link: [MOTFront](http://tiny.cc/MOTFront).

## Requirements
To install network dependencies refer to **environment.yaml**.

## Paths
Store pre-trained networks in the **model** directory.

Store the MOTFront data in the **front_dataset** directory.


## Directories
We split the code into two main blocks, represented with two folders: **Detection** and **Tracking**.

The configurations for training detection backbone network are stored in 

## Basic Usage
For training our end-to-end network, run the command: 
```
python train_combined.py
```

For training the 3D reconstruction and pose estimation pipeline independently, run the command: 
```
python train_net.py
```

For training the tracking pipeline independently, run the command: 
```
python train.py
```

For inference on the 3D reconstruction and pose estimation pipeline, run the command: 
```
python inference_detector.py
```

For inference on our end-to-end network, set the variables **eval_first = True** and **eval_only = True**: 
```
python train_combined.py
```

For inference on our tracking pipeline:
```
python inference.py
```


