import torch
import numpy as np


class ReplayBuffer():
    def __init__(self, input_dims, mem_size, batch_size):
        self.input_dims = input_dims
        self.mem_size = mem_size
        self.batch_size = batch_size
        self.counter = 0

        self.state_memory = np.zeros((self.mem_size, input_dims), dtype=np.float32)
        self.next_state_memory = np.zeros((self.mem_size, input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.done_memory = np.zeros(self.mem_size, dtype=np.bool)


    def save(self, state, action, reward, next_state, done):
        where = self.counter % self.mem_size
        self.state_memory[where] = state 
        self.action_memory[where] = action 
        self.reward_memory[where] = reward
        self.next_state_memory[where] = next_state
        self.done_memory[where] = done 

        self.counter += 1


    def sample(self, device):
        if self.counter < self.batch_size:
            return 
        else:
            max_mem = min(self.mem_size, self.counter)
            batch = np.random.choice(max_mem, self.batch_size, replace=False)
            states = torch.from_numpy(self.state_memory[batch]).to(device)
            actions = torch.from_numpy(self.action_memory[batch]).to(device)
            rewards = torch.from_numpy(self.reward_memory[batch]).to(device)
            next_states = torch.from_numpy(self.next_state_memory[batch]).to(device)
            dones = torch.from_numpy(self.done_memory[batch]).to(device)
            return (states, actions, rewards, next_states, dones)


if __name__ == '__main__':
    buffer = ReplayBuffer(3, 1000, 64)


    
