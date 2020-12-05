import torch
import numpy as np

from agent import DQNAgent
from unityagents import UnityEnvironment

n_episodes = 2000
max_steps = 1000

env = UnityEnvironment(file_name='Banana.app')
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
env_info = env.reset(train_mode=False)[brain_name]

# get agent
agent_param = {
    'gamma': 0.99,
    'epsilon': 1, 
    'lr': 0.0001,
    'state_dims': len(env_info.vector_observations[0]), 
    'n_actions': brain.vector_action_space_size, 
    'hidden_dims': [256, 256],
    'eps_min': 0.01, 
    'eps_decay': 0.995, 
    'mem_size': 100000, 
    'batch_size': 64,
    'tau': 1e-3, 
    'update_every': 4
}

agent = DQNAgent(**agent_param)


def dqn(n_episodes, max_steps):
    scores = list()
    for i in range(1, n_episodes+1):
        env_info = env.reset(train_mode=False)[brain_name]
        state = env_info.vector_observations[0]
        score = 0

        for s in range(max_steps):
            action = agent.act(state)
            env.step(action)[brain_name]
            reward = env_info.rewards[0] 
            next_state = env_info.vector_observations[0]   
            done = env_info.local_done[0]
            agent.step(state, action, reward, next_state, done)
            score += reward
            state = next_state
            if done:
                break

        scores.append(score)
        
        if i % 100 == 0:
            avg_score = np.mean(scores[-100:])
            print(f'Episode {i} \t Past 100 episode has average score {avg_score}')
            if avg_score > 13:
                print(f'Environment solved, achived {avg_score} over 100 episodes')
                torch.save(agent.qnet.state_dict(), 'trained_model.pth')
                break

    return scores


if __name__ == '__main__':
    scores = dqn(n_episodes, max_steps)



