# MQR
Deep reinforcement learning excels at learning control policies in high-dimensional action spaces, making it crucial for robotic manipulation. However, its real-world application is limited by costly and risky data collection. Offline reinforcement learning addresses this issue by training on pre-collected datasets but struggles with Q-value overestimation in high-dimensional discrete action spaces, where the number of out-of-distribution actions rapidly increases, negatively impacting training stability. In this work, we propose the Most overestimated Q-value Regularization (MQR), a novel offline reinforcement learning algorithm that penalizes the action with the most overestimated Q-value, effectively mitigating overestimation in high-dimensional discrete action spaces. By regulating the action most affected by Q-value overestimation, rather than applying uniform penalties across the entire action space as in existing methods, MQR further prevents the policy from converging incorrectly. We evaluate MQR on a robotic pushing and grasping task, a challenging high-dimensional discrete action space problem, in both simulated and real-world environments with random, dense, and unknown object arrangements. The results demonstrate that MQR significantly outperforms baseline algorithms, achieving a clearance rate of 96.94\% in simulations and 99.04\% in real-world dense configurations, while maintaining high action efficiency and stability. These findings highlight MQR’s robustness, scalability, and adaptability for robotic manipulation, showcasing its potential for real-world deployment in industrial robotics.

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



