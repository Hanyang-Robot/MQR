import numpy as np
import time
import dm_env
from dm_control.utils import transformations

from env.camera_real import AzureKinect
from env.robot_real import UR5e
from env.gripper_real import Robotiq2F85

from utils import load_config


config = load_config('conf/test_real.yaml')


class Grasping():
    INITIAL_OBJECT_HEIGHT = 0.2
    PREPOSE_OFFSET = 0.1
    GRASPING_OFFSET = 0.03
    PUSHING_LENGTH = 0.1
    MIN_HEIGHT = 0.01  # default: 0.02

    def __init__(self, config):
        self.heightmap_resolution = config['heightmap_resolution']
        self.num_rotations = config['num_rotations']
        self.workspace_limits = config['robot']['workspace']

        self.robot = UR5e(config)
        self.gripper = Robotiq2F85(config)
        self.camera = AzureKinect(config)

        self.action = None
        self.action_fail_count = [0, 0]
        self.empty_threshold = 10

    def reset(self):
        print('Reset')
        self.robot.movel(self.robot.home_pose)
        self.gripper.activate()
        observation = self.get_observation()
        
        return dm_env.TimeStep(dm_env.StepType.FIRST, None, None, observation)
        
    def step(self, action):
        self._step(action)
        self.after_step()
        reward = self.get_reward()
        observation = self.get_observation()
        episode_over = self.get_termination()
        if episode_over:
            return dm_env.TimeStep(dm_env.StepType.LAST, reward, 1.0, observation)
        else:
            return dm_env.TimeStep(dm_env.StepType.MID, reward, 1.0, observation)
        
    def _step(self, action):
        prepose, target_pose = self.preprocess_action(action)
        
        action_type = action[0]

        if action_type == 'push':
            self.gripper.close()
            self.robot.movel(prepose[0], a=0.7, v=1.2)
            self.collision = self.robot.movel(prepose[1], a=0.2, v=0.2, check_collision=True)
            
            if self.collision:
                self.robot.movel(self.robot.home_pose, a=0.7, v=1.2, check_collision=True)
                self.robot.movel(self.robot.home_pose, a=0.7, v=1.2)
                return
            
            self.robot.movel(target_pose, a=0.1, v=0.2)
            target_pose[2] = prepose[0][2]
            self.robot.movel(target_pose, a=0.7, v=1.2)
        
        elif action_type == 'grasp':
            self.robot.movel(prepose, a=0.7, v=1.2)
            self.collision = self.robot.movel(target_pose, a=0.2, v=0.2, check_collision=True)
            
            if self.collision:
                self.robot.movel(self.robot.home_pose, a=0.7, v=1.2, check_collision=True)
                self.robot.movel(self.robot.home_pose, a=0.7, v=1.2)
                return
            
            self.gripper.close()
            time.sleep(1)
            self.robot.movel(prepose, a=0.7, v=1.2)
            time.sleep(1.5)
            self.grasp_flag = self.gripper.check_grasp()
            
            if self.grasp_flag:
                self.robot.movel(self.robot.placing_pose, a=0.7, v=1.2)
                self.gripper.open()
    
    def after_step(self):
        self.robot.movel(self.robot.home_pose, a=0.7, v=1.2)
        self.gripper.open()
        
    def get_observation(self):
        self.observation = dict()
        color_image, depth_image = self.camera.get_image()
        self.observation['color_image'] = color_image
        self.observation['depth_image'] = depth_image
        color_heightmap, depth_heightmap = self.camera.get_heightmap(self.workspace_limits, self.heightmap_resolution)
        self.observation['color_heightmap'] = color_heightmap
        self.observation['depth_heightmap'] = depth_heightmap

        return self.observation
    
    def get_reward(self):
        self.reward = 0.0
        action_type = self.action[0]

        if action_type == 'push':
            if self.collision:
                print(f'Reward: {self.reward} (push fail: collision)')
                return self.reward

            depth_heightmap = self.observation['depth_heightmap']
            next_color_heightmap, next_depth_heightmap = self.camera.get_heightmap(self.workspace_limits, self.heightmap_resolution)
            change_detected, change_value = self.detect_change(depth_heightmap, next_depth_heightmap)
            if change_detected:
                self.reward = 0.5
                print(f'Reward: {self.reward} (push success | change value: {change_value})')
            else:
                print(f'Reward: {self.reward} (push fail | change value: {change_value})')

        elif action_type == 'grasp':
            if self.collision:
                print(f'Reward: {self.reward} (grasp fail: collision)')
                return self.reward
            if self.grasp_flag:
                self.reward = 1.0
                print(f'Reward: {self.reward} (grasp success)')
            else:
                print(f'Reward: {self.reward} (grasp fail)')     
        
        return self.reward
    
    def get_termination(self):
        # reset flag 1: 10 consecutive action failed
        print(f"### action_fail_count{self.action_fail_count}")
        if self.reward == 0:
            if self.action[0] == 'push':
                self.action_fail_count[0] += 1
            elif self.action[0] == 'grasp':
                self.action_fail_count[1] += 1
        else:
            if self.action[0] == 'push':
                self.action_fail_count[0] = 0
            elif self.action[0] == 'grasp':
                self.action_fail_count[1] = 0

        if sum(self.action_fail_count) >= 10:
            print(f'Reset flag (10 consecutive action failed ({self.action_fail_count}))')
            self.action_fail_count = [0, 0]
            return True

        # reset flag 2: empty workspace
        stuff_count = np.zeros(self.observation['depth_heightmap'].shape)
        stuff_count[self.observation['depth_heightmap'] > 0.001] = 1
        stuff_sum = np.sum(stuff_count)

        if stuff_sum < self.empty_threshold:
            print(f"Not enough objects in view (stuff_count: {stuff_sum})")
            termination = True
        else:
            termination = None

        if termination and not config['is_testing']:
            self.action_fail_count = [0, 0]
            print('Reset flag (empty workspace)')

        return termination        

    
    def preprocess_action(self, action):
        self.action = action # {action_type, rotation_index(ccw), pixel_y, pixel_x, height}
        action_type = action[0]
        rotation_angle = -np.deg2rad(action[1]*(360.0/self.num_rotations)) # ccw
        pixel_y = action[2]
        pixel_x = action[3]
        height = max(self.MIN_HEIGHT, action[4])
        
        # Compute 3D position of pixel, {c} and {w} are symmetric with respect to y = x.
        position = [pixel_y*self.heightmap_resolution + self.workspace_limits[0][0],
                    pixel_x*self.heightmap_resolution + self.workspace_limits[1][0],
                    height]

        pose = transformations.euler_to_rmat([np.pi, 0, rotation_angle], ordering='XYZ', full=True)
        pose[0:3, 3] = position

        if action_type == 'push':
            camera_coordinates_push_direction = np.array([np.cos(rotation_angle), np.sin(rotation_angle)])*self.PUSHING_LENGTH
            
            # Transform push direction from camera coordinates {c} to world coordinates {w}
            world_coordinates_push_direction = self.camera.rotation_matrix @ np.append(camera_coordinates_push_direction, [0, 1])
            
            # If pushing, adjust start position, and make sure z value is safe and not too low
            depth_heightmap = self.observation['depth_heightmap']
            safe_kernel_width = int(np.round((self.gripper.FINGER_WIDTH/2)/self.heightmap_resolution))
            local_region = depth_heightmap[max(pixel_y - safe_kernel_width, 0):min(pixel_y + safe_kernel_width + 1, depth_heightmap.shape[0]),
                                           max(pixel_x - safe_kernel_width, 0):min(pixel_x + safe_kernel_width + 1, depth_heightmap.shape[1])]
            if local_region.size == 0:
                safe_z_position = 0
            else:
                safe_z_position = np.max(local_region)
            pose[2, 3] = safe_z_position
    
            prepose = [None, None]
            prepose[0] = pose @ np.array([[1, 0, 0, 0],
                                          [0, 1, 0, 0],
                                          [0, 0, 1, -self.gripper.CLOSE_LENGTH -self.PREPOSE_OFFSET],
                                          [0, 0, 0, 1]])
            prepose[1] = prepose[0] @ np.array([[1, 0, 0, 0], 
                                                [0, 1, 0, 0],
                                                [0, 0, 1, self.PREPOSE_OFFSET],
                                                [0, 0, 0, 1]])
            
            if prepose[1][2, 3] < self.MIN_HEIGHT + self.gripper.CLOSE_LENGTH:
                prepose[1][2, 3] = self.MIN_HEIGHT + self.gripper.CLOSE_LENGTH
            
            target_pose = np.array([[1, 0, 0, world_coordinates_push_direction[0]],
                                    [0, 1, 0, world_coordinates_push_direction[1]],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]]) @ prepose[1]

            target_pose[0, 3] = np.clip(target_pose[0, 3], self.workspace_limits[0][0], self.workspace_limits[0][1])
            target_pose[1, 3] = np.clip(target_pose[1, 3], self.workspace_limits[1][0], self.workspace_limits[1][1])

            prepose[0] = [prepose[0][0, 3], prepose[0][1, 3], prepose[0][2, 3], np.pi, 0, rotation_angle]
            prepose[1] = [prepose[1][0, 3], prepose[1][1, 3], prepose[1][2, 3], np.pi, 0, rotation_angle]
            target_pose = [target_pose[0, 3], target_pose[1, 3], target_pose[2, 3], np.pi, 0, rotation_angle]

        elif action_type == 'grasp':
            prepose = pose @ np.array([[1, 0, 0, 0],
                                       [0, 1, 0, 0],
                                       [0, 0, 1, -self.gripper.CLOSE_LENGTH - self.PREPOSE_OFFSET],
                                       [0, 0, 0, 1]])
            target_pose = prepose @ np.array([[1, 0, 0, 0],
                                             [0, 1, 0, 0],
                                             [0, 0, 1, self.PREPOSE_OFFSET + self.GRASPING_OFFSET],
                                             [0, 0, 0, 1]])

            if target_pose[2, 3] < self.MIN_HEIGHT + self.gripper.CLOSE_LENGTH:
                target_pose[2, 3] = self.MIN_HEIGHT + self.gripper.CLOSE_LENGTH

            prepose = [prepose[0, 3], prepose[1, 3], prepose[2, 3], np.pi, 0, rotation_angle]
            target_pose = [target_pose[0, 3], target_pose[1, 3], target_pose[2, 3], np.pi, 0, rotation_angle]
        
        else:
            raise Exception('invalid action type')

        return prepose, target_pose

    def detect_change(self, depth_heightmap, next_depth_heightmap):
        depth_difference = abs(next_depth_heightmap - depth_heightmap)
        depth_difference[np.isnan(depth_difference)] = 0
        depth_difference[depth_difference > 0.05] = 0
        depth_difference[depth_difference < 0.005] = 0
        depth_difference[depth_difference > 0] = 1
        change_value = np.sum(depth_difference)
        change_threshold = 20
        if change_value > change_threshold:
            change_detected = True
        else:
            change_detected = False

        return change_detected, change_value