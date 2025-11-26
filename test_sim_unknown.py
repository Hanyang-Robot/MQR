import numpy as np
import agent
import env
from utils import load_config, set_seed

config = load_config('conf/test_sim_unknown.yaml')
set_seed(config['seed'])

# --- Define Evaluation Metrics ---------------------
num_clear_objects = 0       # Completion Ratio
total_action_steps = 0      # Action Efficiency Ratio
grasp_steps = 0             # Grasp Success Ratio
push_steps = 0
push_success_steps = 0  
num_objects = config['num_objects']
# ---------------------------------------------------

env = env.load(config)
agent = agent.load(config)
agent.load_model(config['model_weight'])

time_step = env.reset() # {step_type, reward, discount, observation}
next_state = time_step.observation

while len(agent.clearance_log) < config['max_test_steps']:
    print('-'*70, f'\nStep: {agent.step} (episode: {len(agent.clearance_log)})')
    
    state = next_state
    action = agent.get_action(state) # {action_type, rotation_index(ccw), pixel_y, pixel_x, height}
    time_step = env.step(action)
    reward = time_step.reward
    next_state = time_step.observation

    agent.train(state, action, reward, next_state)
    agent.step += 1

    # --- Evalutate performance --------------------
    if action[0] == 'push':
        push_steps += 1
        total_action_steps += 1
        if reward == 0.5:
            push_success_steps += 1

    elif action[0] == 'grasp':
        grasp_steps += 1
        total_action_steps += 1
        if reward == 1:
            num_clear_objects += 1
    # ----------------------------------------------

    if time_step.last():
        if grasp_steps==0: grasp_steps+=0.0001
        completion_ratio = np.round((num_clear_objects / num_objects)*100, 2)
        grasp_success_ratio = np.round((num_clear_objects / grasp_steps)*100, 2)

        if push_steps==0: push_success_ratio=-1
        else: push_success_ratio = np.round((push_success_steps / push_steps)*100, 2)
        action_efficiency_ratio = np.round((num_clear_objects / total_action_steps)*100, 2)
        print(f"completion: {completion_ratio}%, grasp success: {grasp_success_ratio}%, push success: {push_success_ratio}%, action efficiency: {action_efficiency_ratio}%,\npush steps: {push_steps}, grasp steps: {grasp_steps}\n")

        break