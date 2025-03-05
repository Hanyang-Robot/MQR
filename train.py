import time
import agent
import env
from utils import load_config, set_seed

config = load_config('conf/train.yaml')

set_seed(config['seed'])

agent = agent.load(config)
agent.preload(transitions_directory=agent.logger.transitions_directory)

while agent.step < config['max_train_steps']:
    print('-'*70, f'\nStep: {agent.step}')
    iteration_time_0 = time.time()
    
    agent.train_offline()
    agent.step += 1

    iteration_time_1 = time.time()
    exe_time = iteration_time_1-iteration_time_0
    print(f'Time elapsed: {exe_time:.3f}') 

    if agent.step % 1000 == 0:
        agent.logger.save_model(agent.step, agent.model)