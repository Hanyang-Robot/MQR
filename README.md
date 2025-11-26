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
* Collect an offline dataset in a simulation environment.
  ```shell
  python generate_dataset_sim.py
  ```
  * The offline dataset collected in the simulation is stored in the "MQR/logs/offline_dataset_sim/" directory.
  * The data folder contains state information (RGB-D heightmap), while the models folder stores the parameters of the online VPG model.
  * In addition, the transitions folder includes the action (executed_action.txt) and reward (reward_value.txt) data.
  * If you want to modify the settings for offline dataset collection, please refer to the "generate_dataset_sim.yaml" file in the conf directory.
  * The image below illustrates the process of collecting an offline dataset in the simulation environment.
    <img width="1015" height="762" alt="Image" src="https://github.com/user-attachments/assets/9cb1ad91-f916-4072-b662-599438f5ec0c" />
  
* Collect an offline dataset in a real environment.
  ```shell
  python generate_dataset_real.py
  ```
  * The offline dataset collected in the real-world is stored in the "MQR/logs/offline_dataset_real/" directory.
  * If you want to modify the settings for offline dataset collection, please refer to the "generate_dataset_real.yaml" file in the conf directory.
 
* You can download the offline dataset files from the Google Drive link: 

# :three: Training offline RL policy
```shell
python train.py
```
* If you want to modify the settings for training offline RL policy, please refer to the "train.yaml" file in the conf directory.
* You must write the path to the offline dataset in the "log_directory" variable of the train.yaml file.

# :four: Testing offline RL policy
### Testing policy on the simulation environment with random arrangements.
```shell
python test_sim_random.py
```
* You can download the weight files (MQR_sim.pth & MQR_real.pth) from the Google Drive link: https://drive.google.com/file/d/1-8dPlDZBAUrFACLcOj5VpgNwMYH-ySlO/view?usp=drive_link
* If you want to modify the settings for testing sim policy, please refer to the "test_sim_random.yaml" file in the conf directory.
* You must write the path to the weight file for offline RL policy in the "model_weight" variable of the test_sim_random.yaml file.

### Testing policy on the simulation environment with dense arrangement.
```shell
python test_sim_dense.py
```
* If you want to modify the settings for testing sim_dense policy, please refer to the "test_sim_dense.yaml" file in the conf directory.
* You must write the path to the weight file for offline RL policy in the "model_weight" variable of the test_sim_dense.yaml file.

### Testing policy on the simulation environment with unknown objects.
```shell
python test_sim_unknown.py
```
* If you want to modify the settings for testing sim_dense policy, please refer to the "test_sim_unknown.yaml" file in the conf directory.
* You must write the path to the weight file for offline RL policy in the "model_weight" variable of the test_sim_unknown.yaml file.

### Testing policy on the real environment.
```shell
python test_real.py
```
* If you want to modify the settings for testing real policy, please refer to the "test_real.yaml" file in the conf directory.
* You must write the path to the weight file for offline RL policy in the "model_weight" variable of the test_real.yaml file.





