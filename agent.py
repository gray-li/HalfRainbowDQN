import torch
import numpy as np
import torch.nn.functional as F
from torch import nn, optim

from qnet import DeepQNetwork
from replay_buffer import ReplayBuffer


class Agent():
    def __init__(self, gamma, epsilon, lr, state_dims, n_actions, hidden_dims,
                eps_min=0.1, eps_decay=0.99, mem_size=100000, batch_size=128,
                tau=1e-3, update_every=4):
        self.gamma = gamma
        self.epsilon = epsilon 
        self.lr = lr 
        self.state_dims = state_dims
        self.n_actions = n_actions
        self.hidden_dims = hidden_dims
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.tau = tau
        self.update_every = update_every
        self.step_counter = 0
        self.action_space = [i for i in range(self.n_actions)]
        self.buffer = ReplayBuffer(self.input_dims, self.mem_size, self.batch_size)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.qnet = DeepQNetwork(self.state_dims, self.n_actions, 
                                self.hidden_dims, self.lr)

        self.target_qnet = DeepQNetwork(self.state_dims, self.n_actions, 
                                        self.hidden_dims, self.lr)

        self.criterion =  F.mse_loss()
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=self.lr)


    def act(self, state):
        if np.random.random() >  self.epsilon:
            state = torch.from_numpy(state).float().unsqueeze(0).to(self.qnet.device)
            self.qnet.eval()
            with torch.no_grad():
                qs = self.qnet(state)
            self.qnet.train()
            action = np.argmax(qs.item())
        else:
            action = np.random.choice(self.action_space)

        return action


    def learn(self):
        if self.buffer.counter < self.buffer.batch_size:
            return 

        states, actions, rewards, next_states, dones = self.buffer.sample(self.qnet.device)

        q_expected = self.qnet(states).gather(1, actions)

        q_target_next = self.target_qnet(next_states).detach().max(1)[0].unsqueeze(1)
        q_target = rewards + (self.gamma * q_target_next * (1 - dones))

        loss = self.criterion(q_expected, q_target)
        self.qnet.optimizer.zero_grad()
        loss.backward()
        self.qnet.optimizer.step()

        self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)

        self.update_weights()


    def update_weights(self):
        for target_weights, qnet_weights in zip(self.target_qnet.parameters(), self.qnet.parameters()):
            value = self.tau * qnet_weights.data + (1.0 - self.tau) * target_weights.data
            target_weights.data.copy_(value)


    def step(self, state, action, reward, next_state, done):
        self.buffer.save(state, action, reward, next_state, done)

        if self.step_counter % self.update_every == 0:
            self.learn()

        self.step_counter += 1





    




        
        


        

