import agent
import env
from utils import load_config, set_seed

config = load_config('conf/test.yaml')

set_seed(config['seed'])

env = env.load(config)
agent = agent.load(config)
agent.load_model(config['model_weight'])

time_step = env.reset() # {step_type, reward, discount, observation}
next_state = time_step.observation

while len(agent.clearance_log) < config['max_test_episodes']:
    print('-'*70, f'\nStep: {agent.step} (episode: {len(agent.clearance_log)})')
    
    state = next_state
    action = agent.get_action(state) # {action_type, rotation_index(ccw), pixel_y, pixel_x, height}
    time_step = env.step(action)
    reward = time_step.reward
    next_state = time_step.observation

    agent.train(state, action, reward, next_state)

    agent.step += 1

    if time_step.last():
        time_step = env.reset()
        next_state = time_step.observation
        agent.clearance_log.append([agent.step - 1])
        agent.logger.log('clearance', agent.clearance_log)
        agent.load_model(config['model_weight'])