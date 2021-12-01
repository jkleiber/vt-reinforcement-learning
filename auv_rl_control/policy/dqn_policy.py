from policy.policy import Policy

from math import radians
import numpy as np

# PyTorch
import torch
from torch import nn
from torch import optim
from torch.distributions import Normal

class DQNPolicy(Policy):

    def __init__(self, action_map, n_states = 5, n_actions = 49, n_hidden = 128, lr = 1e-5):
        # Map action index to action value
        self.action_map = action_map
        
        # Get input/output dimensions
        self.state_dim = n_states
        self.action_dim = n_actions

        # Hidden layer dimension
        self.hidden_dim = n_hidden

        # Learning rate
        self.learn_rate = lr

        # Epsilon
        self.epsilon = 1.0 # Reduce as time goes on

        # Main network
        self.main_network = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.action_dim)
        )

        # Target network
        self.target_network = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.action_dim)
        )

        self.main_optimizer = optim.Adam(self.main_network.parameters(), lr=self.learn_rate)

        # Experience replay
        self.replay_buffer = []


    def get_action(self, state):
        # Epsilon greedy
        a = torch.rand(1)

        best_val_idx = 0
        if a < self.epsilon:
            random_value_fn = torch.rand(self.state_dim)
            best_val_idx = torch.argmax(random_value_fn).int()
        else:
            # Convert the state into a tensor
            state_t = torch.from_numpy(state)

            # Estimate q function
            q_function = self.main_network(state_t)
            best_val_idx = torch.argmax(q_function).int()

        return self.action_map[best_val_idx]

    def update(self):
        pass