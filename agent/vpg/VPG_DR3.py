import os

import cv2
import numpy as np
import torch
import torch.optim as optim
from scipy import ndimage

from .logger import Logger
from .logger_RMO import Logger_for_RMO
from .model_DR3 import reinforcement_net
from .utils import visualize_q_maps


class DR3():
    def __init__(self, config):
        self.future_reward_discount = config['future_reward_discount']
        self.save_visualization = config['save_visualization']
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {self.device}')

        # Fully convolutional Q-network for deep reinforcement learning
        self.model = reinforcement_net(config).to(self.device)
        self.target_model = reinforcement_net(config).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())

        # Initialize Huber loss and optimizer
        self.criterion = torch.nn.SmoothL1Loss(reduction='none').to(self.device)

        self.is_testing = config['is_testing']
        if self.is_testing:
            self.logger = Logger(config)
            self.optimizer = optim.SGD(self.model.parameters(), lr=1e-5, momentum=0.9, weight_decay=2e-5)
        else:
            self.logger = Logger_for_PROMPT(config)
            self.optimizer = optim.SGD(self.model.parameters(), lr=1e-4, momentum=0.9, weight_decay=2e-5)

        # Set model to training mode
        self.model.train()

        # Initialize lists to save execution info and RL variables
        self.step = 0
        self.is_exploit_log = []
        self.executed_action_log = []
        self.reward_value_log = []
        self.q_value_log = []
        self.label_value_log = []
        self.loss_value_log = []
        self.per_loss_value_log = []
        self.clearance_log = []

        self.offline_loss_total_log = []
        self.ofline_loss_TD_log = []
        self.offline_loss_CQL_log = []
        self.offline_loss_DR3_log = []

        self.c_0 = 0.03
        self.cql_alpha = 1.0
        self.dataset_size = 1.0
        self.method = config['method']

    # Compute forward pass through model to compute Q-maps
    def forward(self, color_heightmap, depth_heightmap, is_volatile=False, specific_rotation=-1, use_target=False):
        # Apply 2x scale to input heightmaps
        color_heightmap_2x = ndimage.zoom(color_heightmap, zoom=[2,2,1], order=0)
        depth_heightmap_2x = ndimage.zoom(depth_heightmap, zoom=[2,2], order=0)
        assert(color_heightmap_2x.shape[0:2] == depth_heightmap_2x.shape[0:2])

        # Add extra padding (to handle rotations inside network)
        diagonal_length = float(color_heightmap_2x.shape[0]) * np.sqrt(2)
        diagonal_length = np.ceil(diagonal_length/32)*32
        padding_width = int((diagonal_length - color_heightmap_2x.shape[0])/2)
        color_heightmap_2x_r =  np.pad(color_heightmap_2x[:, :, 0], padding_width, 'constant', constant_values=0)
        color_heightmap_2x_r.shape = (color_heightmap_2x_r.shape[0], color_heightmap_2x_r.shape[1], 1)
        color_heightmap_2x_g =  np.pad(color_heightmap_2x[:, :, 1], padding_width, 'constant', constant_values=0)
        color_heightmap_2x_g.shape = (color_heightmap_2x_g.shape[0], color_heightmap_2x_g.shape[1], 1)
        color_heightmap_2x_b =  np.pad(color_heightmap_2x[:, :, 2], padding_width, 'constant', constant_values=0)
        color_heightmap_2x_b.shape = (color_heightmap_2x_b.shape[0], color_heightmap_2x_b.shape[1], 1)
        color_heightmap_2x = np.concatenate((color_heightmap_2x_r, color_heightmap_2x_g, color_heightmap_2x_b), axis=2)
        depth_heightmap_2x =  np.pad(depth_heightmap_2x, padding_width, 'constant', constant_values=0)
        
        # Pre-process color image (scale and normalize)
        image_mean = [0.485, 0.456, 0.406]
        image_std = [0.229, 0.224, 0.225]
        input_color_image = color_heightmap_2x.astype(float)/255
        for channel in range(3):
            input_color_image[:, :, channel] = (input_color_image[:, :, channel] - image_mean[channel])/image_std[channel]

        # Pre-process depth image (normalize)
        image_mean = [0.01, 0.01, 0.01]
        image_std = [0.03, 0.03, 0.03]
        depth_heightmap_2x.shape = (depth_heightmap_2x.shape[0], depth_heightmap_2x.shape[1], 1)
        input_depth_image = np.concatenate((depth_heightmap_2x, depth_heightmap_2x, depth_heightmap_2x), axis=2)
        for channel in range(3):
            input_depth_image[:, :, channel] = (input_depth_image[:, :, channel] - image_mean[channel])/image_std[channel]

        # Construct minibatch of size 1 (b,c,h,w)
        input_color_image.shape = (input_color_image.shape[0], input_color_image.shape[1], input_color_image.shape[2], 1)
        input_depth_image.shape = (input_depth_image.shape[0], input_depth_image.shape[1], input_depth_image.shape[2], 1)
        input_color_data = torch.from_numpy(input_color_image.astype(np.float32)).permute(3,2,0,1)
        input_depth_data = torch.from_numpy(input_depth_image.astype(np.float32)).permute(3,2,0,1)
        
        # Pass input data through model
        if use_target:
            output_prob, state_feat, push_feat, grasp_feat = self.target_model.forward(input_color_data, input_depth_data, is_volatile, specific_rotation)
        else:
            output_prob, state_feat, push_feat, grasp_feat = self.model.forward(input_color_data, input_depth_data, is_volatile, specific_rotation)
        
        # Return Q-maps (and remove extra padding)
        for rotation_index in range(len(output_prob)):
            if rotation_index == 0:
                push_q_maps = output_prob[rotation_index][0].cpu().data.numpy()[:, 0, int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2), int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2)]
                grasp_q_maps = output_prob[rotation_index][1].cpu().data.numpy()[:, 0, int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2), int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2)]
            else:
                push_q_maps = np.concatenate((push_q_maps, output_prob[rotation_index][0].cpu().data.numpy()[:, 0, int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2), int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2)]), axis=0)
                grasp_q_maps = np.concatenate((grasp_q_maps, output_prob[rotation_index][1].cpu().data.numpy()[:, 0, int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2), int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2)]), axis=0)

        return push_q_maps, grasp_q_maps, state_feat, push_feat, grasp_feat

    def get_action(self, state):
        color_heightmap = state['color_heightmap']
        depth_heightmap = state['depth_heightmap']

        # Save RGB-D images and RGB-D heightmaps
        self.logger.save_images(self.step, state['color_image'], state['depth_image'])
        self.logger.save_heightmaps(self.step, state['color_heightmap'], state['depth_heightmap'])

        # Run forward pass with network to get affordances
        push_q_maps, grasp_q_maps, state_feat, _, _ = self.forward(color_heightmap, depth_heightmap, is_volatile=True)

        # Determine whether grasping or pushing should be executed based on network predictions
        best_push_q_value = np.max(push_q_maps)
        best_grasp_q_value = np.max(grasp_q_maps)
        print(f'Q-values: {best_push_q_value:.3f} (push), {best_grasp_q_value:.3f} (grasp)')
        if best_push_q_value > best_grasp_q_value:
            action_type = 'push'
        else:
            action_type = 'grasp'
        print(f'Action type: {action_type}')

        # Get pixel location and rotation with highest Q-value (rotation, y, x)
        if action_type == 'push':
            best_pixel_index = np.unravel_index(np.argmax(push_q_maps), push_q_maps.shape)
            q_value = best_push_q_value
        else:
            best_pixel_index = np.unravel_index(np.argmax(grasp_q_maps), grasp_q_maps.shape)
            q_value = best_grasp_q_value

        # Save predicted confidence value
        self.q_value_log.append([q_value])
        self.logger.log('q_value', self.q_value_log)

        # Save executed action
        if action_type == 'push':
            self.executed_action_log.append([0, best_pixel_index[0], best_pixel_index[1], best_pixel_index[2]]) # 0: push
        elif action_type == 'grasp':
            self.executed_action_log.append([1, best_pixel_index[0], best_pixel_index[1], best_pixel_index[2]]) # 1: grasp
        self.logger.log('executed_action', self.executed_action_log)

        # Visualize executed action and affordances
        if self.save_visualization:
            push_q_maps_image = visualize_q_maps(push_q_maps, color_heightmap, best_pixel_index)
            self.logger.save_visualization(self.step, push_q_maps_image, 'push')
            grasp_q_maps__image = visualize_q_maps(grasp_q_maps, color_heightmap, best_pixel_index)
            self.logger.save_visualization(self.step, grasp_q_maps__image, 'grasp')

        best_rotation_index = best_pixel_index[0]
        best_pixel_y = best_pixel_index[1]
        best_pixel_x = best_pixel_index[2]
        height = depth_heightmap[best_pixel_y][best_pixel_x]

        action = [action_type, best_rotation_index, best_pixel_y, best_pixel_x, height]
        
        return action

    def train(self, state, action, reward, next_sate):
        color_heightmap = state['color_heightmap']
        depth_heightmap = state['depth_heightmap']
        action_type = action[0]
        best_pix_index = action[1:]
        next_color_heightmap = next_sate['color_heightmap']
        next_depth_heightmap = next_sate['depth_heightmap']

        # Compute training labels
        label_value, _ = self.get_label_value(reward, next_color_heightmap, next_depth_heightmap)
        self.label_value_log.append([label_value])
        self.logger.log('label_value', self.label_value_log)
        self.reward_value_log.append([reward])
        self.logger.log('reward_value', self.reward_value_log)

        # Backpropagate
        loss_value = self.backpropagate(color_heightmap, depth_heightmap, action_type, best_pix_index, label_value)
        self.loss_value_log.append([loss_value])
        self.logger.log('loss_value', self.loss_value_log)
        
        # Update target network weight with soft update
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(0.05*param.data + (1 - 0.05)*target_param.data)

    def train_offline(self):
        size_index = int(len(self.q_value_log)*self.dataset_size)
        print(f"size_index : {size_index}")
        tmp_sample_surprise_values = np.abs(np.asarray(self.q_value_log[0:size_index]) - np.asarray(self.label_value_log[0:size_index]))

        # Initialize sample values
        for idx_sample in range(len(tmp_sample_surprise_values)):
            if tmp_sample_surprise_values[idx_sample] == 0:
                tmp_sample_surprise_values[idx_sample] = 1000

        # Preventing convergence to a probability of 0.
        sample_surprise_values = tmp_sample_surprise_values + 0.001

        # Sorting TD values
        sorted_surprise_ind = np.argsort(sample_surprise_values[:,0])

        # Sample Index (As the surprise value increases, the probability of being sampled also increases)
        pow_law_exp = 2
        rand_sample_ind = int(np.round(np.random.power(pow_law_exp, 1)*(sorted_surprise_ind.size-1)))
        sample_step = sorted_surprise_ind[rand_sample_ind]
        print(f'Experience replay: step {sample_step} (surprise value: {sample_surprise_values[sorted_surprise_ind[rand_sample_ind]][0]:.3f})')

        # Load sample RGB-D heightmap
        sample_color_heightmap = cv2.imread(os.path.join(self.logger.color_heightmaps_directory, f'{sample_step:06d}.color.png'))
        sample_depth_heightmap = cv2.imread(os.path.join(self.logger.depth_heightmaps_directory, f'{sample_step:06d}.depth.png'), -1)
        sample_depth_heightmap = sample_depth_heightmap.astype(np.float32)/100000

        # Load next sample RGB-D heightmap
        next_sample_color_heightmap = cv2.imread(os.path.join(self.logger.color_heightmaps_directory, f'{sample_step+1:06d}.color.png'))
        next_sample_depth_heightmap = cv2.imread(os.path.join(self.logger.depth_heightmaps_directory, f'{sample_step+1:06d}.depth.png'), -1)
        next_sample_depth_heightmap = next_sample_depth_heightmap.astype(np.float32)/100000     

        # Load action
        if self.executed_action_log[sample_step][0] == 0: sample_action_type = 'push'
        elif self.executed_action_log[sample_step][0] == 1: sample_action_type = 'grasp'
        sample_best_pix_index = (np.asarray(self.executed_action_log)[sample_step, 1:4]).astype(int)

        # Load reward
        sample_reward = self.reward_value_log[sample_step][0]   

        # Compute forward pass with sample
        with torch.no_grad():
            sample_push_q_maps, sample_grasp_q_maps, sample_state_feat, _, _ = self.forward(sample_color_heightmap, sample_depth_heightmap, is_volatile=True)

        # Get labels for sample and backpropagate
        label_value, next_feat = self.get_label_value(sample_reward, next_sample_color_heightmap, next_sample_depth_heightmap)
        loss_total, loss_TD, loss_CQL, loss_DR3 = self.backpropagate_offline(sample_color_heightmap, sample_depth_heightmap, sample_action_type, sample_best_pix_index, label_value, next_feat)
        
        self.offline_loss_total_log.append([loss_total])
        self.logger.log('offline_loss_total', self.offline_loss_total_log)
        self.ofline_loss_TD_log.append([loss_TD])
        self.logger.log('offline_loss_TD', self.ofline_loss_TD_log)
        self.offline_loss_CQL_log.append([loss_CQL])
        self.logger.log('offline_loss_CQL', self.offline_loss_CQL_log)
        self.offline_loss_DR3_log.append([loss_DR3])
        self.logger.log('offline_loss_DR3', self.offline_loss_DR3_log)

        # Recompute prediction value and label for replay buffer
        if sample_action_type == 'push':
            self.q_value_log[sample_step] = [np.max(sample_push_q_maps)]
        elif sample_action_type == 'grasp':
            self.q_value_log[sample_step] = [np.max(sample_grasp_q_maps)]

        self.label_value_log.append([label_value])
        self.logger.log('label_value', self.label_value_log)

        # Update target network weight with soft update
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(0.05*param.data + (1 - 0.05)*target_param.data)

    def get_label_value(self, reward, next_color_heightmap, next_depth_heightmap):
        next_push_q_maps, next_grasp_q_maps, next_state_feat, next_push_feat, next_grasp_feat = \
            self.forward(next_color_heightmap, next_depth_heightmap, is_volatile=True, use_target=True)

        max_next_push_q = np.max(next_push_q_maps)
        max_nest_grasp_q = np.max(next_grasp_q_maps)

        if max_next_push_q > max_nest_grasp_q:
            next_push_idx = np.unravel_index(np.argmax(next_push_q_maps), next_push_q_maps.shape)
            next_feat = next_push_feat[next_push_idx[0]]
        else:
            next_grasp_idx = np.unravel_index(np.argmax(next_grasp_q_maps), next_grasp_q_maps.shape)
            next_feat = next_grasp_feat[next_grasp_idx[0]]

        # Compute future reward
        if reward == 0:
            next_q_value = 0
            
        else:
            next_q_value = max(max_next_push_q, max_nest_grasp_q)

        td_target = reward + self.future_reward_discount * next_q_value
        print(f'TD-target value: {reward} + {self.future_reward_discount} x {next_q_value:.3f} = {td_target:.3f}')
        return td_target, next_feat

    # TODO: use Q-maps from get_action
    # Compute labels and backpropagate
    def backpropagate(self, color_heightmap, depth_heightmap, action_type, best_pix_index, label_value):
        # Compute labels
        label = np.zeros((1, 320, 320))
        action_area = np.zeros((224, 224))
        action_area[best_pix_index[1]][best_pix_index[2]] = 1
        tmp_label = np.zeros((224, 224))
        tmp_label[action_area > 0] = label_value
        label[0, 48:(320 - 48), 48:(320 - 48)] = tmp_label

        # Compute label mask
        label_weights = np.zeros(label.shape)
        tmp_label_weights = np.zeros((224, 224))
        tmp_label_weights[action_area > 0] = 1
        label_weights[0, 48:(320 - 48), 48:(320 - 48)] = tmp_label_weights
        
        # Compute loss and backward pass
        self.optimizer.zero_grad()
        loss_value = 0
        if action_type == 'push':
            # Do forward pass with specified rotation (to save gradients)
            push_q_map, grasp_q_map, state_feat, _, _ = self.forward(color_heightmap, depth_heightmap, specific_rotation=best_pix_index[0])
            loss = self.criterion(self.model.output_prob[0][0].view(1, 320, 320), torch.from_numpy(label).float().to(self.device)) * torch.from_numpy(label_weights).float().to(self.device)
            loss = loss.sum()
            loss.backward()
            loss_value = loss.cpu().data.numpy()

        elif action_type == 'grasp':
            # Do forward pass with specified rotation (to save gradients)
            push_q_map, grasp_q_map, state_feat, _, _ = self.forward(color_heightmap, depth_heightmap, specific_rotation=best_pix_index[0])
            loss = self.criterion(self.model.output_prob[0][1].view(1, 320, 320), torch.from_numpy(label).float().to(self.device)) * torch.from_numpy(label_weights).float().to(self.device)
            loss = loss.sum()
            loss.backward()
            loss_value = loss.cpu().data.numpy()

            # Since grasping is symmetric, train with another forward pass of opposite rotation angle
            opposite_rotate_idx = (best_pix_index[0] + self.model.num_rotations/2) % self.model.num_rotations
            push_q_map, grasp_q_map, state_feat, _, _ = self.forward(color_heightmap, depth_heightmap, specific_rotation=opposite_rotate_idx)
            loss = self.criterion(self.model.output_prob[0][1].view(1, 320, 320), torch.from_numpy(label).float().to(self.device)) * torch.from_numpy(label_weights).float().to(self.device)
            loss = loss.sum()
            loss.backward()
            loss_value += loss.cpu().data.numpy()

            loss_value = loss_value/2

        print(f'Training loss: {loss_value:.3f}')
        self.optimizer.step()

        return loss_value

    # (Offline Reinforcement Learning) Compute labels and backpropagate
    def backpropagate_offline(self, color_heightmap, depth_heightmap, action_type, best_pix_index, label_value, next_feat):
        # Compute labels
        label = np.zeros((1, 320, 320))
        action_area = np.zeros((224, 224))
        action_area[best_pix_index[1]][best_pix_index[2]] = 1
        tmp_label = np.zeros((224, 224))
        tmp_label[action_area > 0] = label_value
        label[0, 48:(320 - 48), 48:(320 - 48)] = tmp_label

        # Compute label mask
        label_weights = np.zeros(label.shape)
        tmp_label_weights = np.zeros((224, 224))
        tmp_label_weights[action_area > 0] = 1
        label_weights[0, 48:(320 - 48), 48:(320 - 48)] = tmp_label_weights
        
        # Compute loss and backward pass
        self.optimizer.zero_grad()
        loss_total = 0
        if action_type == 'push':
            # Do forward pass with specified rotation (to save gradients)
            push_q_map, grasp_q_map, state_feat, push_feat, grasp_feat = self.forward(color_heightmap, depth_heightmap, specific_rotation=best_pix_index[0])
            
            # Compute data Q-value for CQL Loss
            data_Q_map = self.model.output_prob[0][0].view(1,320,320)[0,48:(320-48),48:(320-48)]
            data_Q_value = data_Q_map[best_pix_index[1]][best_pix_index[2]]

            # Compute loss fo TD
            loss_TD = self.criterion(self.model.output_prob[0][0].view(1, 320, 320), torch.from_numpy(label).float().to(self.device)) * torch.from_numpy(label_weights).float().to(self.device)
            loss_TD = loss_TD.sum()

            # Compute loss for CQL
            loss_CQL = self.compute_cql_loss(data_Q_map, data_Q_value)

            # Compute loss for DR3
            tmp_feat_vector = push_feat[0].view(-1)
            feat_vector = tmp_feat_vector.t()
            next_feate_vector = next_feat[0].view(-1)
            cosine_similarity = abs(torch.sum(torch.dot(feat_vector, next_feate_vector)))
            loss_DR3 = self.c_0 * cosine_similarity

            # Backward
            loss_total = loss_TD + loss_CQL + loss_DR3
            loss_total.backward()

            # For logging
            loss_TD = loss_TD.cpu().data.numpy()
            loss_CQL = loss_CQL.cpu().data.numpy()
            loss_DR3 = loss_DR3.cpu().data.numpy()
            loss_total = loss_total.cpu().data.numpy()

        elif action_type == 'grasp':
            # Do forward pass with specified rotation (to save gradients)
            push_q_map, grasp_q_map, state_feat, push_feat, grasp_feat = self.forward(color_heightmap, depth_heightmap, specific_rotation=best_pix_index[0])
            
            # Compute data Q-value for CQL Loss
            data_Q_map = self.model.output_prob[0][1].view(1,320,320)[0,48:(320-48),48:(320-48)]
            data_Q_value = data_Q_map[best_pix_index[1]][best_pix_index[2]]

            # Compute loss fo TD
            loss_TD_p = self.criterion(self.model.output_prob[0][1].view(1, 320, 320), torch.from_numpy(label).float().to(self.device)) * torch.from_numpy(label_weights).float().to(self.device)
            loss_TD_p = loss_TD_p.sum()

            # Compute loss for CQL
            loss_CQL_p = self.compute_cql_loss(data_Q_map, data_Q_value)

            # Compute loss for DR3
            tmp_feat_vector = grasp_feat[0].view(-1)
            feat_vector = tmp_feat_vector.t()
            next_feate_vector = next_feat[0].view(-1)
            cosine_similarity = abs(torch.sum(torch.dot(feat_vector, next_feate_vector)))
            loss_DR3_p = self.c_0 * cosine_similarity

            # Backward
            loss_total_p = loss_TD_p + loss_CQL_p + loss_DR3_p
            loss_total_p.backward()

            # For logging
            loss_TD_p = loss_TD_p.cpu().data.numpy()
            loss_CQL_p = loss_CQL_p.cpu().data.numpy() 
            loss_DR3_p = loss_DR3_p.cpu().data.numpy() 
            loss_total_p = loss_total_p.cpu().data.numpy()

            #===== Since grasping is symmetric, train with another forward pass of opposite rotation angle =====#
            opposite_rotate_idx = (best_pix_index[0] + self.model.num_rotations/2) % self.model.num_rotations
            push_q_map, grasp_q_map, state_feat, push_feat, grasp_feat = self.forward(color_heightmap, depth_heightmap, specific_rotation=opposite_rotate_idx)
            
            # Compute data Q-value for CQL Loss
            data_Q_map = self.model.output_prob[0][1].view(1,320,320)[0,48:(320-48),48:(320-48)]
            data_Q_value = data_Q_map[best_pix_index[1]][best_pix_index[2]]

            # Compute loss for TD
            loss_TD_n = self.criterion(self.model.output_prob[0][1].view(1, 320, 320), torch.from_numpy(label).float().to(self.device)) * torch.from_numpy(label_weights).float().to(self.device)
            loss_TD_n = loss_TD_n.sum()

            # Compute loss for CQL
            loss_CQL_n = self.compute_cql_loss(data_Q_map, data_Q_value)

            # Compute loss for DR3
            tmp_feat_vector = grasp_feat[0].view(-1)
            feat_vector = tmp_feat_vector.t()
            next_feate_vector = next_feat[0].view(-1)
            cosine_similarity = abs(torch.sum(torch.dot(feat_vector, next_feate_vector)))
            loss_DR3_n = self.c_0 * cosine_similarity

            # Backward
            loss_total_n = loss_TD_n + loss_CQL_n + loss_DR3_n
            loss_total_n.backward()

            # For logging
            loss_TD_n = loss_TD_n.cpu().data.numpy()
            loss_CQL_n = loss_CQL_n.cpu().data.numpy() 
            loss_DR3_n = loss_DR3_n.cpu().data.numpy() 
            loss_total_n = loss_total_n.cpu().data.numpy()

            loss_TD = (loss_TD_p+loss_TD_n)/2
            loss_CQL = (loss_CQL_p+loss_CQL_n)/2
            loss_DR3 = (loss_DR3_p+loss_DR3_n)/2
            loss_total = (loss_total_p+loss_total_n)/2

        print(f'Training loss: {loss_total:.3f}, DR3 loss: {loss_DR3:.3f}, CQL loss: {loss_CQL:.3f}, TD loss: {loss_TD:.3f}')
        self.optimizer.step()

        return loss_total, loss_TD, loss_CQL, loss_DR3

    # Load pre-trained model weight
    def load_model(self, model_weight):
        self.model.load_state_dict(torch.load(model_weight))
        print(f'Load trained model: {model_weight}')

    # Loss function of PROMPT
    def compute_cql_loss(self, Q_maps, data_Q_value):
        """Computes the CQL loss for a batch of Q-values and actions."""
        if self.method == 'sum':
            exp_Q_maps = torch.exp(Q_maps)
            sum_exp_Q_maps = torch.sum(exp_Q_maps)
            tmp_logsumexp_Q_maps = torch.log(sum_exp_Q_maps)
            logsumexp_Q_maps = tmp_logsumexp_Q_maps * self.cql_alpha 
            _loss_cql = logsumexp_Q_maps - data_Q_value
            print(f"log-sum-exp-Q : {logsumexp_Q_maps:.3f}, data-Q : {data_Q_value:.3f}")

            return _loss_cql  

        elif self.method == 'max':
            tmp_cql_loss = torch.max(Q_maps)
            alpha_cql_loss = tmp_cql_loss * self.cql_alpha 
            _loss_cql = alpha_cql_loss - data_Q_value
            print(f"max-Q : {alpha_cql_loss:.3f}, data-Q : {data_Q_value:.3f}")

            return _loss_cql

        elif self.method == 'min':
            tmp_q_flatten = torch.flatten(Q_maps)
            q_flatten = tmp_q_flatten[tmp_q_flatten > 0]

            # To prevent the value from becoming empty.
            if len(q_flatten) == 0: q_flatten = torch.tensor([1e-6])

            tmp_cql_loss = torch.min(q_flatten)
            alpha_cql_loss = tmp_cql_loss * self.cql_alpha 
            _loss_cql = alpha_cql_loss - data_Q_value
            print(f"min-Q : {alpha_cql_loss:.3f}, data-Q : {data_Q_value:.3f}")

            return _loss_cql  

    # Pre-load execution info and RL variables
    def preload(self, transitions_directory):
        print(f"transitions_directory : {transitions_directory}")

        self.executed_action_log = np.loadtxt(os.path.join(transitions_directory, 'executed_action.txt'), delimiter=' ')
        self.iteration = self.executed_action_log.shape[0]-1
        self.executed_action_log = self.executed_action_log[0:self.iteration,:]
        self.executed_action_log = self.executed_action_log.tolist()

        self.q_value_log = np.loadtxt(os.path.join(transitions_directory, 'q_value.txt'), delimiter=' ')
        self.q_value_log = self.q_value_log[0:self.iteration]
        self.q_value_log.shape = (self.iteration,1)
        self.q_value_log = self.q_value_log.tolist()

        self.reward_value_log = np.loadtxt(os.path.join(transitions_directory, 'reward_value.txt'), delimiter=' ')
        self.reward_value_log = self.reward_value_log[0:self.iteration]
        self.reward_value_log.shape = (self.iteration,1)
        self.reward_value_log = self.reward_value_log.tolist()

        self.label_value_log = np.loadtxt(os.path.join(transitions_directory, 'label_value.txt'), delimiter=' ')
        self.label_value_log = self.label_value_log[0:self.iteration]
        self.label_value_log.shape = (self.iteration,1)
        self.label_value_log = self.label_value_log.tolist()