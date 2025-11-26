import datetime
import os
import time

import cv2
import numpy as np
import torch
import yaml


class Logger():
    def __init__(self, config):
        print(f'Agent: {config["agent"]}')
        print(f'Object type: {config["object_type"]}')
        # print(f'Num objects: {config["num_objects"]}')

        # Create directory to save data
        log_directory = config['log_directory']

        if config['is_continue']:
            data_directory = config['data_directory']
            self.base_directory = os.path.join(log_directory, data_directory)
            self.color_images_directory = os.path.join(self.base_directory, 'data', 'color_images')
            self.depth_images_directory = os.path.join(self.base_directory, 'data', 'depth_images')
            self.color_heightmaps_directory = os.path.join(self.base_directory, 'data', 'color_heightmaps')
            self.depth_heightmaps_directory = os.path.join(self.base_directory, 'data', 'depth_heightmaps')
            self.models_directory = os.path.join(self.base_directory, 'models')
            self.visualizations_directory = os.path.join(self.base_directory, 'visualizations')
            self.transitions_directory = os.path.join(self.base_directory, 'transitions')

        else:
            timestamp_value = datetime.datetime.fromtimestamp(time.time())
            self.base_directory = os.path.join(log_directory, timestamp_value.strftime('%Y_%m_%d_%H_%M_%S'))
            print(f'Creating data logging session: {self.base_directory}')
            self.color_images_directory = os.path.join(self.base_directory, 'data', 'color_images')
            os.makedirs(self.color_images_directory, exist_ok=True)
            self.depth_images_directory = os.path.join(self.base_directory, 'data', 'depth_images')
            os.makedirs(self.depth_images_directory, exist_ok=True)
            self.color_heightmaps_directory = os.path.join(self.base_directory, 'data', 'color_heightmaps')
            os.makedirs(self.color_heightmaps_directory, exist_ok=True)
            self.depth_heightmaps_directory = os.path.join(self.base_directory, 'data', 'depth_heightmaps')
            os.makedirs(self.depth_heightmaps_directory, exist_ok=True)
            self.models_directory = os.path.join(self.base_directory, 'models')
            os.makedirs(self.models_directory, exist_ok=True)
            self.visualizations_directory = os.path.join(self.base_directory, 'visualizations')
            os.makedirs(self.visualizations_directory, exist_ok=True)
            self.transitions_directory = os.path.join(self.base_directory, 'transitions')
            os.makedirs(self.transitions_directory, exist_ok=True)

        # Save config
        output_filepath = os.path.join(self.base_directory, 'config.yaml')
        with open(output_filepath, 'w') as file:
            yaml.dump(config, file)

    def save_images(self, step, color_image, depth_image):
        cv2.imwrite(os.path.join(self.color_images_directory, f'{step:06d}.color.png'), color_image)
        depth_image = np.round(depth_image * 10000).astype(np.uint16) # Save depth in 1e-4 meters
        cv2.imwrite(os.path.join(self.depth_images_directory, f'{step:06d}.depth.png'), depth_image)
    
    def save_heightmaps(self, step, color_heightmap, depth_heightmap):
        cv2.imwrite(os.path.join(self.color_heightmaps_directory, f'{step:06d}.color.png'), color_heightmap)
        depth_heightmap = np.round(depth_heightmap * 100000).astype(np.uint16) # Save depth in 1e-5 meters
        cv2.imwrite(os.path.join(self.depth_heightmaps_directory, f'{step:06d}.depth.png'), depth_heightmap)
    
    def log(self, log_name, log):
        np.savetxt(os.path.join(self.transitions_directory, f'{log_name}.txt'), log, delimiter=' ')

    def save_model(self, step, model):
        torch.save(model.state_dict(), os.path.join(self.models_directory, f'model_{step:06d}.pth'))

    def save_model_backup(self, model):
            torch.save(model.state_dict(), os.path.join(self.models_directory, f'model_backup.pth'))

    def save_visualization(self, step, q_maps_vis, name):
        cv2.imwrite(os.path.join(self.visualizations_directory, f'{step:06d}_{name}.png'), q_maps_vis)