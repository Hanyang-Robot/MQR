import time
import agent
import env
from utils import load_config, set_seed

config = load_config('conf/generate_dataset_real.yaml')
set_seed(config['seed'])

env = env.load(config)
agent = agent.load(config)

if config['is_continue']:
    agent.load_model('/model_backup.pth')
    agent.preload()
else:
    agent.load_model_sim_weights('/home/cai-tp/PROMPT/logs/online_VPG_050000.pth')

time_step = env.reset() # {step_type, reward, discount, observation}
next_state = time_step.observation

while agent.step < config['max_train_steps']:
    print('-'*70, f'\nStep: {agent.step}')
    iteration_time_0 = time.time()
    
    state = next_state
    action = agent.get_action(state) # {action_type, rotation_index(ccw), pixel_y, pixel_x, height}
    print(f"action_type: {action[0]}, rotation_index(ccw): {action[1]}, pixel_y: {action[2]}, pixel_x: {action[3]}, height: {action[4]}")
    time_step = env.step(action)
    reward = time_step.reward
    next_state = time_step.observation

    agent.train(state, action, reward, next_state)
    agent.step += 1

    iteration_time_1 = time.time()
    print('Time elapsed: %f' % (iteration_time_1-iteration_time_0)) 

    if time_step.last():
        # Is continue?
        if config['environment_type']=='real':
            key_p = input("Please respawn the objects in the workspace. Once the respawn is complete, press 'p'. If you want to end the training, please press 'f'.")
            if key_p == 'f':
                break
            else:
                while key_p != 'p':
                    key_p = input("To continue, enter 'p':")

        time.sleep(2)
        time_step = env.reset()
        next_state = time_step.observation
        agent.clearance_log.append([agent.step - 1])
        agent.logger.log('clearance', agent.clearance_log)

    # save model per 1000 steps
    if agent.step % 100 == 0:
        agent.logger.save_model(agent.step, agent.model)

    # save model for backup
    agent.logger.save_model_backup(agent.model)