from dm_control.rl import control

from env import mjcf_sim
from env import mjcf_sim_dense
from env import mjcf_real
from env import env_sim
from env import env_sim_dense
from env import env_real

def load(config, num_objects=None):
    if config ['environment_type'] == 'sim':
        if config['env_dense'] == False:
            mjcf_model = mjcf_sim.load(config)
            physics = env_sim.Physics.from_mjcf_model(mjcf_model)
            task = env_sim.Grasping(physics, config)

        elif config['env_dense'] == True:
            mjcf_model = mjcf_sim_dense.load(config, num_objects)
            physics = env_sim_dense.Physics.from_mjcf_model(mjcf_model)
            task = env_sim_dense.Grasping(physics, config, num_objects)

    elif config ['environment_type'] == 'real':  
        mjcf_model = mjcf_real.load(config)     
        return env_real.Grasping(config)

    return control.Environment(physics, task)