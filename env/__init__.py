from dm_control.rl import control

from env import mjcf
from env import mjcf_dense
from env import mjcf_real
from env import env_vpg
from env import env_vpg_dense
from env import grasping_vpg_real

def load(config, num_objects=None):
    if config ['environment_type'] == 'sim':
        if config['is_dense_arrangements'] == False:
            mjcf_model = mjcf.load(config)
            physics = env_vpg.Physics.from_mjcf_model(mjcf_model)
            task = env_vpg.Grasping(physics, config)

        elif config['is_dense_arrangements'] == True:
            mjcf_model = mjcf_dense.load(config, num_objects)
            physics = env_vpg_dense.Physics.from_mjcf_model(mjcf_model)
            task = env_vpg_dense.Grasping(physics, config, num_objects)

    elif config ['environment_type'] == 'real':  
        mjcf_model = mjcf_real.load(config)     
        return grasping_vpg_real.Grasping(config)

    return control.Environment(physics, task)