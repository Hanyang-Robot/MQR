import os
import random

import matplotlib.colors as mcolors
import numpy as np
from dm_control import mjcf

def load(config):
    # Scene
    scene_xml_path = os.path.join(os.getcwd(), 'env', 'assets', 'scene.xml')
    scene = mjcf.from_path(scene_xml_path)

    # Robot
    robot_xml_path = os.path.join(os.getcwd(), 'env', 'assets', 'universal_robots_ur5e', 'ur5e.xml')
    robot = mjcf.from_path(robot_xml_path)
    scene.attach(robot)

    # Gripper
    gripper_xml_path = os.path.join(os.getcwd(), 'env', 'assets', 'robotiq_2f85', '2f85.xml')
    gripper = mjcf.from_path(gripper_xml_path)
    attachment_site = robot.find('site', 'attachment_site')
    attachment_site.attach(gripper)

    # Grasping area
    scene.worldbody.add('site',
                        name='grasping_area',
                        type='box',
                        rgba=[0, 0, 0, 1],
                        size=[0.224, 0.224, 1e-3], 
                        pos=[0.45, 0, 0])
    
    # Placing area
    scene.worldbody.add('site',
                        name='placing_area',
                        type='box',
                        rgba=[255, 255, 255, 1],
                        size=[0.224, 0.224, 1e-3], 
                        pos=[3, 0, 0])

    # Camera
    # scene.worldbody.add('camera',
    #                     name='front_view',
    #                     pos=[1.5, 0, 1],
    #                     euler=[0, np.pi/3, np.pi/2])
    # scene.worldbody.add('camera',
    #                     name='top_down_view',
    #                     fovy=config['camera']['fovy'],
    #                     pos=[0.45, 0, 0.9],
    #                     euler=[0, 0, np.pi/2])

    # Object
    # https://matplotlib.org/stable/gallery/color/named_colors.html#tableau-palette
    tableau_palette = mcolors.to_rgba_array(mcolors.TABLEAU_COLORS)

    object_file_dir = os.path.join(os.getcwd(), 'env', 'assets', config['object_type'])
    object_name_list = os.listdir(object_file_dir)
    for id in range(config['num_objects']):
        # object_name = random.choice(object_name_list)
        object_name = object_name_list[id%len(object_name_list)]
        file_names = os.listdir(os.path.join(object_file_dir, object_name))
        xml_file_name = [file_name for file_name in file_names if file_name.endswith('.xml')][0]
        object_xml_path = os.path.join(object_file_dir, object_name, xml_file_name)
        object = mjcf.from_path(object_xml_path)
        object.model = f'object_{id}'
        object.default.geom.condim = 6
        object.default.geom.solref = [0.01, 1]
        object.default.geom.solimp = [0.95, 0.99, 0.0001]

        object.default.geom.rgba = tableau_palette[id%10]

        meshes = object.find_all('mesh')
        for mesh in meshes:
            mesh.scale = [0.772]*3 # robotiq gripper stroke(85mm)/on-robot gripper stroke(110mm)
        
        attachment_frame = scene.attach(object)
        attachment_frame.pos = [5 + (id//5)*0.4, (id%5)*0.4, 0.4]
        attachment_frame.add('freejoint')

    # Options
    scene.option.integrator = 'implicit'
    scene.option.cone = 'elliptic'
    scene.option.flag.multiccd = 'enable'

    return scene