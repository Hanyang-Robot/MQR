# MQR
We propose the Most overestimated Q-value Regularization (MQR), a novel offline reinforcement learning algorithm that penalizes the action with the most overestimated Q-value, effectively mitigating overestimation in high-dimensional discrete action spaces. This repository provides mujoco and pytorch code for training and testing MQR.

### The Most Overestimated Q-value Regularization in High-dimensional Discrete Action Spaces for Offline Reinforcement Learing
Paper: This paper is scheduled to be released through the official IEEE Transactions on Neural Networks and Learning Systems (T-NNLS).
* The Overview of the proposed MQR framework
<img width="2062" height="701" alt="Image" src="https://github.com/user-attachments/assets/61e14ed9-504c-4d02-9541-6eed3272a030" />

* Example for robotic pushing and grasping
<img width="4256" height="2359" alt="Image" src="https://github.com/user-attachments/assets/5896d3d2-0c53-438d-b8a7-01f8d78f2d8e" />

## Installation

This code has been tested with python 3.9, pytorch 1.12.0, NVIDIA DRIVER 470, CUDA 11.3, cuDNN 8.2.1 on Ubuntu 20.04.6 LTS.

Create the conda env and install necessary python libraries.
```
conda env create -f mqr.yaml
```

## Inference
```shell
python train.py
```







