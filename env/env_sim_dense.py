import cv2
import numpy as np
from dm_control import mujoco
from dm_control import mjcf
from dm_control.rl import control
from dm_control.utils import transformations

from env.camera_sim import Camera
from env.robot_sim import UR5e
from env.gripper_sim import Robotiq2F85
from env.utils import sample_pose


class Physics(mjcf.Physics):
    def step_with_rendering(self, iteration, nstep, width=1024, height=768, camera_id='front_view', rendering=True):
        for _ in range(iteration):
            self.step(nstep=nstep)
            if rendering:
                frame = self.render(height, width, camera_id)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cv2.imshow(camera_id, frame)
                cv2.waitKey(1)

    def check_contact(self, geoms_1, geoms_2):
        for i in range(self.data.ncon):
            c1_in_g1 = self.data.contact.geom1[i] in geoms_1
            c2_in_g2 = self.data.contact.geom2[i] in geoms_2
            c2_in_g1 = self.data.contact.geom2[i] in geoms_1
            c1_in_g2 = self.data.contact.geom1[i] in geoms_2
            if (c1_in_g1 and c2_in_g2) or (c1_in_g2 and c2_in_g1):
                return True
        return False


class Grasping(control.Task):
    INITIAL_OBJECT_HEIGHT = 0.2
    PREPOSE_OFFSET = 0.1
    GRASPING_OFFSET = 0.03
    PUSHING_LENGTH = 0.1
    MIN_HEIGHT = 0.02

    def __init__(self, physics, config, num_objects):
        self.is_check_collision = False

        self.num_objects = num_objects
        self.heightmap_resolution = config['heightmap_resolution']
        self.num_rotations = config['num_rotations']

        self.robot = UR5e()
        self.gripper = Robotiq2F85(robot_name=self.robot.NAME)
        self.camera = Camera(physics, config, 'top_down_view')
        self.camera_matrices = self.camera.matrices()

        names = physics.model.names.decode('utf-8').split('\x00')
        self.gripper_pad_geoms = [physics.model.name2id(name, 'geom') for name in self.gripper.pad_names]
        
        # Object geoms
        object_geom_names = [name for name in names if 'object' in name and 'unnamed_geom' in name]
        self.object_geoms = [[] for _ in range(self.num_objects)]
        for i in range(self.num_objects):
            self.object_geoms[i] = [physics.model.name2id(name, 'geom') for name in object_geom_names if f'object_{i}/' in name]
        
        self.grasping_area_position = physics.named.model.site_pos['grasping_area']
        self.grasping_area_size = physics.named.model.site_size['grasping_area']/2
        self.grasping_area_size_for_check = physics.named.model.site_size['grasping_area']*1.2
        self.grasping_area = [(self.grasping_area_position[i] - self.grasping_area_size[i] + self.gripper.FINGER_WIDTH,
                              self.grasping_area_position[i] + self.grasping_area_size[i] - self.gripper.FINGER_WIDTH) for i in range(2)]
        self.grasping_area_for_termination = [(self.grasping_area_position[i] - self.grasping_area_size[i]*2 + self.gripper.FINGER_WIDTH,
                                            self.grasping_area_position[i] + self.grasping_area_size[i]*2 - self.gripper.FINGER_WIDTH) for i in range(2)]
        self.grasping_area_for_check = [(self.grasping_area_position[i] - self.grasping_area_size_for_check[i] + self.gripper.FINGER_WIDTH,
                              self.grasping_area_position[i] + self.grasping_area_size_for_check[i] - self.gripper.FINGER_WIDTH) for i in range(2)]
        
        self.placing_area_position = physics.named.model.site_pos['placing_area']
        self.placing_area_size = physics.named.model.site_size['placing_area']
        self.placing_area_size_for_check = physics.named.model.site_size['placing_area']*1.5
        self.placing_area = [(self.placing_area_position[i] - self.placing_area_size[i],
                              self.placing_area_position[i] + self.placing_area_size[i]) for i in range(2)]
        self.placing_area_for_check = [(self.placing_area_position[i] - self.placing_area_size_for_check[i],
                              self.placing_area_position[i] + self.placing_area_size_for_check[i]) for i in range(2)]

        self.workspace = [(self.grasping_area_position[i] - self.grasping_area_size[i]*2,
                           self.grasping_area_position[i] + self.grasping_area_size[i]*2) for i in range(2)]

        self.num_of_inside_objs = 0
        self.num_of_outside_objs_by_grasp = 0
        self.num_of_outside_objs_by_push = 0
        self.list_of_outside_objs_by_grasp = np.array([])
        self.list_of_outside_objs_by_push = np.array([])

    def initialize_episode(self, physics):
        print('-'*70, '\nReset')
        self.robot.set(physics, self.robot.HOME_JOINT_POSITION)
        self.gripper.set(physics, self.gripper.OPEN_POSITION)
        self.sample_object(physics)
        self.action_fail_count = [0, 0]

        physics.step_with_rendering(iteration=100, nstep=5)

    def before_step(self, action, physics):
        target_pose, prepose, success = self.preprocess_action(physics, action)

        if not success:
            self.grasped_object_index = None
            return
        
        action_type = action[0]

        if action_type == 'push':
            self.gripper.close(physics)
            self.robot.set(physics, self.robot.INITIAL_JOINT_POSITION)
            self.robot.set(physics, prepose[0])
            self.grasped_object_index = None

            # check collision between robot and object
            if self.is_check_collision:
                collision = self.robot.movel(physics, prepose[1], check_collision=True, geoms_1=self.gripper_pad_geoms, geoms_2=sum(self.object_geoms, []))                
                if not collision: 
                    self.robot.movel(physics, target_pose)

            else:
                self.robot.movel(physics, prepose[1])
                self.robot.movel(physics, target_pose)

        elif action_type == 'grasp':
            self.robot.set(physics, self.robot.INITIAL_JOINT_POSITION)
            self.robot.set(physics, prepose)

            # check collision between robot and object
            if self.is_check_collision:
                collision = self.robot.movel(physics, target_pose, check_collision=True, geoms_1=self.gripper_pad_geoms, geoms_2=sum(self.object_geoms, []))
                if not collision:
                    self.gripper.close(physics)
                    self.robot.movel(physics, prepose)
                    self.grasped_object_index = self.check_grasp(physics)
                    if self.grasped_object_index is not None:
                        self.move_grasped_object(physics)

                else:
                    print(f"collision occurs between robot and objects!!")
                    self.grasped_object_index = None

            else:
                self.robot.movel(physics, target_pose)
                self.gripper.close(physics)
                self.robot.movel(physics, prepose)
                self.grasped_object_index = self.check_grasp(physics)
                if self.grasped_object_index is not None:
                    self.move_grasped_object(physics)

    def after_step(self, physics):
        if self.grasped_object_index is not None:
            self.move_grasped_object(physics)
        self.robot.set(physics, self.robot.HOME_JOINT_POSITION)
        self.gripper.set(physics, self.gripper.OPEN_POSITION)
        
        physics.step_with_rendering(iteration=100, nstep=5)

    def action_spec(self, physics):
        return mujoco.action_spec(physics)

    def get_observation(self, physics):
        self.observation = dict()
        self.observation['color_image'] = self.camera.get_color_image(physics) # RGB
        self.observation['depth_image'] = self.camera.get_depth_image(physics, noise=False)
        color_heightmap, depth_heightmap = self.camera.get_heightmap(physics, self.workspace, self.heightmap_resolution, noise=False)
        self.observation['color_heightmap'] = color_heightmap
        self.observation['depth_heightmap'] = depth_heightmap

        return self.observation
    
    def get_reward(self, physics):
        self.reward = 0.0
        action_type = self.action[0]

        if action_type == 'push':
            depth_heightmap = self.observation['depth_heightmap']
            next_color_heightmap, next_depth_heightmap = self.camera.get_heightmap(physics, self.workspace, self.heightmap_resolution, noise=False)
            change_detected= self.detect_change(depth_heightmap, next_depth_heightmap)
            if change_detected:
                self.reward = 0.5
                print(f'Reward: {self.reward} (push success)')
            else:
                print(f'Reward: {self.reward} (push fail)')
        
        elif action_type == 'grasp':
            if self.grasped_object_index is not None:
                self.reward = 1.0
                print(f'Reward: {self.reward} (grasp success: object {self.grasped_object_index})')
            else:
                print(f'Reward: {self.reward} (grasp fail)')
        
        return self.reward
    
    def get_termination(self, physics):
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

            return True

        for object_id in range(self.num_objects):
            object_position = physics.named.data.qpos[f'object_{object_id}/'][0:3].copy()
            if ((self.grasping_area_for_termination[0][0] < object_position[0] < self.grasping_area_for_termination[0][1]) and
                (self.grasping_area_for_termination[1][0] < object_position[1] < self.grasping_area_for_termination[1][1])):
                termination = None
                break
            else:
                termination = True

        if termination:
            print('Reset flag (empty workspace)')

        return termination

    def sample_object(self, physics):
        for object_id in range(self.num_objects):
            self.place_object_in_grasping_area(physics, object_id)

        for object_id in range(self.num_objects):
            object_position = physics.named.data.qpos[f'object_{object_id}/'][0:3].copy()
            if not((self.grasping_area[0][0] < object_position[0] < self.grasping_area[0][1]) and
                   (self.grasping_area[1][0] < object_position[1] < self.grasping_area[1][1])):
                self.place_object_in_grasping_area(physics, object_id)

    def place_object_in_grasping_area(self, physics, object_id):
        while True:
            random_pose = sample_pose(self.grasping_area[0], self.grasping_area[1], [self.INITIAL_OBJECT_HEIGHT]*2)
            physics.named.data.qpos[f'object_{object_id}/'] = random_pose
            
            previous_object_position = physics.named.data.qpos[f'object_{object_id}/'][0:3].copy()
            physics.step_with_rendering(iteration=100, nstep=5)
            current_object_position = physics.named.data.qpos[f'object_{object_id}/'][0:3].copy()
            
            while np.linalg.norm(previous_object_position - current_object_position) > 1e-3:
                previous_object_position = current_object_position
                physics.step_with_rendering(iteration=1, nstep=5)
                current_object_position = physics.named.data.qpos[f'object_{object_id}/'][0:3].copy()

            if ((self.grasping_area[0][0] < current_object_position[0] < self.grasping_area[0][1]) and
                (self.grasping_area[1][0] < current_object_position[1] < self.grasping_area[1][1])):
                break
    
    def preprocess_action(self, physics, action):
        self.action = action # {action_type, rotation_index(ccw), pixel_y, pixel_x, height}
        action_type = self.action[0]
        rotation_angle = -np.deg2rad(self.action[1]*(360.0/self.num_rotations)) # ccw
        pixel_y = self.action[2]
        pixel_x = self.action[3]
        height = self.action[4]
        
        # Compute 3D position of pixel, {c} and {w} are symmetric with respect to y = x.
        position = [pixel_y*self.heightmap_resolution + self.workspace[0][0],
                    pixel_x*self.heightmap_resolution + self.workspace[1][0],
                    height]

        pose = transformations.euler_to_rmat([np.pi, 0, rotation_angle], ordering='XYZ', full=True)
        pose[0:3, 3] = position

        if action_type == 'push':
            camera_coordinates_push_direction = np.array([np.cos(rotation_angle), np.sin(rotation_angle)])*self.PUSHING_LENGTH
            
            # Transform push direction from camera coordinates {c} to world coordinates {w}, R_wc = R_wc' @ R_c'c
            transformation = self.camera_matrices.rotation @ self.camera.OPTICAL_AXIS_ROTATION
            world_coordinates_push_direction = transformation @ np.append(camera_coordinates_push_direction, [0, 1])
            
            # If pushing, adjust start position, and make sure z value is safe and not too low
            depth_heightmap = self.observation['depth_heightmap']
            safe_kernel_width = int(np.round((self.gripper.FINGER_WIDTH/2)/self.heightmap_resolution))
            local_region = depth_heightmap[max(pixel_y - safe_kernel_width, 0):min(pixel_y + safe_kernel_width + 1, depth_heightmap.shape[0]),
                                           max(pixel_x - safe_kernel_width, 0):min(pixel_x + safe_kernel_width + 1, depth_heightmap.shape[1])]
            if not(local_region.size == 0):
                safe_z_position = np.max(local_region)
                pose[2, 3] = safe_z_position
    
            if pose[2, 3] < self.grasping_area_size[2] + self.MIN_HEIGHT:
                pose[2, 3] = self.grasping_area_size[2] + self.MIN_HEIGHT

            prepose = [None, None]
            prepose[0] = pose @ np.array([[1, 0, 0, 0],
                                          [0, 1, 0, 0],
                                          [0, 0, 1, -self.gripper.CLOSE_LENGTH -self.PREPOSE_OFFSET],
                                          [0, 0, 0, 1]])
            prepose[1] = prepose[0] @ np.array([[1, 0, 0, 0], 
                                                [0, 1, 0, 0],
                                                [0, 0, 1, self.PREPOSE_OFFSET],
                                                [0, 0, 0, 1]])
            target_pose = np.array([[1, 0, 0, world_coordinates_push_direction[0]],
                                    [0, 1, 0, world_coordinates_push_direction[1]],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]]) @ prepose[1]
            
            prepose_0_ik_result = self.robot.inverse_kinematics(physics, prepose[0])
            prepose_1_ik_result = self.robot.inverse_kinematics(physics, prepose[1])
            target_pose_ik_result = self.robot.inverse_kinematics(physics, target_pose)
            success = prepose_0_ik_result.success and prepose_1_ik_result.success and target_pose_ik_result.success

        elif action_type == 'grasp':
            prepose = pose @ np.array([[1, 0, 0, 0],
                                       [0, 1, 0, 0],
                                       [0, 0, 1, -self.gripper.CLOSE_LENGTH - self.PREPOSE_OFFSET],
                                       [0, 0, 0, 1]])
            target_pose = prepose @ np.array([[1, 0, 0, 0],
                                             [0, 1, 0, 0],
                                             [0, 0, 1, self.PREPOSE_OFFSET + self.GRASPING_OFFSET],
                                             [0, 0, 0, 1]])

            if target_pose[2, 3] < self.gripper.CLOSE_LENGTH + self.grasping_area_size[2] + self.MIN_HEIGHT:
                target_pose[2, 3] = self.gripper.CLOSE_LENGTH + self.grasping_area_size[2] + self.MIN_HEIGHT
            
            prepose_ik_result = self.robot.inverse_kinematics(physics, prepose)
            target_pose_ik_result = self.robot.inverse_kinematics(physics, target_pose)
            success = prepose_ik_result.success and target_pose_ik_result.success
        
        else:
            raise Exception("Invalid action type")
        
        if not success:
            print('Grasp is kinematically infeasible.')

        return target_pose, prepose, success
    
    def check_grasp(self, physics):
        for index in range(self.num_objects):
            contact = physics.check_contact(self.gripper_pad_geoms, self.object_geoms[index])
            if contact:
                return index
        return None

    def move_grasped_object(self, physics):
        random_pose = sample_pose(self.placing_area[0], self.placing_area[1], [self.INITIAL_OBJECT_HEIGHT]*2)
        physics.named.data.qpos[f'object_{self.grasped_object_index}/'] = random_pose
        physics.step_with_rendering(iteration=1, nstep=5)
    
    def detect_change(self, depth_heightmap, next_depth_heightmap):
        depth_difference = abs(next_depth_heightmap - depth_heightmap)
        depth_difference[np.isnan(depth_difference)] = 0
        depth_difference[depth_difference > 0.3] = 0
        depth_difference[depth_difference < 0.03] = 0
        depth_difference[depth_difference > 0] = 1
        change_value = np.sum(depth_difference)
        change_threshold = 400
        if change_value > change_threshold:
            change_detected = True
        else:
            change_detected = False

        return change_detected

    