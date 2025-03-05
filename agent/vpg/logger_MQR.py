import datetime
import os
import time

import cv2
import numpy as np
import torch
import yaml


class Logger_for_MQR():
    def __init__(self, config, random_seed=None):
        print(f'Agent: {config["agent"]}')

        self.base_directory = config['log_directory']
        self.color_images_directory = os.path.join(self.base_directory, 'data', 'color_images')
        self.depth_images_directory = os.path.join(self.base_directory, 'data', 'depth_images')
        self.color_heightmaps_directory = os.path.join(self.base_directory, 'data', 'color_heightmaps')
        self.depth_heightmaps_directory = os.path.join(self.base_directory, 'data', 'depth_heightmaps')
        self.models_directory = os.path.join(self.base_directory, 'models')
        os.makedirs(self.models_directory, exist_ok=True)
        self.visualizations_directory = os.path.join(self.base_directory, 'visualizations')
        self.transitions_directory = os.path.join(self.base_directory, 'transitions')
        
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

    def save_visualization(self, step, q_maps_vis, name):
        cv2.imwrite(os.path.join(self.visualizations_directory, f'{step:06d}_{name}.png'), q_maps_vis)