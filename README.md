# MQR
We propose the Most overestimated Q-value Regularization (MQR), a novel offline reinforcement learning algorithm that penalizes the action with the most overestimated Q-value, effectively mitigating overestimation in high-dimensional discrete action spaces. This repository provides mujoco and pytorch code for training and testing MQR.

## The Most Overestimated Q-value Regularization in High-dimensional Discrete Action Spaces for Offline Reinforcement Learing
Paper: This paper is scheduled to be released through the official IEEE Transactions on Neural Networks and Learning Systems (T-NNLS).
* The Overview of the proposed MQR framework
  <img width="2062" height="701" alt="Image" src="https://github.com/user-attachments/assets/61e14ed9-504c-4d02-9541-6eed3272a030" />

* For example, robotic pushing and grasping tasks require precise decision-making within high-dimensional discrete action spaces.
  <img width="4256" height="2359" alt="Image" src="https://github.com/user-attachments/assets/5896d3d2-0c53-438d-b8a7-01f8d78f2d8e" />

# :one: Installation

This code has been tested with python 3.9, pytorch 1.12.0, NVIDIA DRIVER 470, CUDA 11.3, cuDNN 8.2.1 on Ubuntu 20.04.6 LTS.

Create the conda env and install necessary python libraries.
```shell
conda env create -f mqr.yaml
```

# :two: Collection Offline Dataset
* You can download the offline dataset files (11.18 GB) from the Google Drive link:
  https://drive.google.com/file/d/1I0putLuqAi4DDrPeQHdzo8l-_EERKMWD/view?usp=sharing
## Collect an offline dataset in a simulation environment.
```shell
python generate_dataset_sim.py
```
* The offline dataset collected in the simulation is stored in the "MQR/logs/offline_dataset_sim/" directory.
* The data folder contains state information (RGB-D heightmap), while the models folder stores the parameters of the online VPG model.
* In addition, the transitions folder includes the action (executed_action.txt) and reward (reward_value.txt) data.
* If you want to modify the settings for offline dataset collection, please refer to the "generate_dataset_sim.yaml" file in the conf directory.
* The image below illustrates the process of collecting an offline dataset in the simulation environment.
  
  <img width="1015" height="762" alt="Image" src="https://github.com/user-attachments/assets/9cb1ad91-f916-4072-b662-599438f5ec0c" />
  
## Collect an offline dataset in a real environment.
```shell
python generate_dataset_real.py
```
* The offline dataset collected in the real-world is stored in the "MQR/logs/offline_dataset_real/" directory.
* If you want to modify the settings for offline dataset collection, please refer to the "generate_dataset_real.yaml" file in the conf directory.

# :three: Training offline RL policy
```shell
python train.py
```
* If you want to modify the settings for training offline RL policy, please refer to the "train.yaml" file in the conf directory.
* You must write the path to the offline dataset in the "log_directory" variable of the train.yaml file.

# :four: Testing offline RL policy
* You can download the weight files (233.6 MB) from the Google Drive link:  
  https://drive.google.com/file/d/1_XPfSjmlC970fYzVtDx41ZBjn-56v2oJ/view?usp=sharing
## Testing policy on the simulation environment with random arrangements.
```shell
python test_sim_random.py
```
* If you want to modify the settings for testing sim policy, please refer to the "test_sim_random.yaml" file in the conf directory.
* You must write the path to the weight file for offline RL policy in the "model_weight" variable of the test_sim_random.yaml file.
* Below is an example video for test_sim_random.

  ![Image](https://github.com/user-attachments/assets/ebb07e8e-f3de-41b6-b8b5-0980665e7505)

## Testing policy on the simulation environment with dense arrangement.
```shell
python test_sim_dense.py
```
* If you want to modify the settings for testing sim_dense policy, please refer to the "test_sim_dense.yaml" file in the conf directory.
* You must write the path to the weight file for offline RL policy in the "model_weight" variable of the test_sim_dense.yaml file.
* Below is an example video for test_sim_dense.

  ![Image](https://github.com/user-attachments/assets/8e412621-77b8-4373-b69e-0f306f36ef31)

## Testing policy on the simulation environment with unknown objects.
```shell
python test_sim_unknown.py
```
* If you want to modify the settings for testing sim_dense policy, please refer to the "test_sim_unknown.yaml" file in the conf directory.
* You must write the path to the weight file for offline RL policy in the "model_weight" variable of the test_sim_unknown.yaml file.
* Below is an example video for test_sim_unknown.

  ![Image](https://github.com/user-attachments/assets/62df7de9-afca-40d1-8f00-94169a76a0bd)

## Testing policy on the real environment.
```shell
python test_real.py
```
* If you want to modify the settings for testing real policy, please refer to the "test_real.yaml" file in the conf directory.
* You must write the path to the weight file for offline RL policy in the "model_weight" variable of the test_real.yaml file.
* Below is an example video for test_real_dense

  ![Image](https://github.com/user-attachments/assets/cf849ecd-3fc5-4950-83c3-e73b0f3ce246)

* Below is an example video for test_real_challenge

  ![Image](https://github.com/user-attachments/assets/b0d770a7-7ca2-4ec0-b86d-6417816d7ae7)

* Below is an example video for test_real_unknown

  ![Image](https://github.com/user-attachments/assets/76b7f37d-aa24-4d53-aad9-be25e3eb8ff9)

  



















