# MQR
We propose the Most overestimated Q-value Regularization (MQR), a novel offline reinforcement learning algorithm that penalizes the action with the most overestimated Q-value, effectively mitigating overestimation in high-dimensional discrete action spaces. By regulating the action most affected by Q-value overestimation, rather than applying uniform penalties across the entire action space as in existing methods, MQR further prevents the policy from converging incorrectly. This repository provides mujoco and pytorch code for training and testing MQR.

### The Most Overestimated Q-value Regularization in High-dimensional Discrete Action Spaces for Offline Reinforcement Learing

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




