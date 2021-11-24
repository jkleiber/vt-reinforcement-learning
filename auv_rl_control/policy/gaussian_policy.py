
from policy.policy import Policy

from math import radians
import numpy as np

# PyTorch
import torch
from torch import nn
from torch import optim
from torch.distributions import Normal

class GaussianPolicy(Policy):

    def __init__(self, n_states = 5, n_actions = 2, n_hidden = 128):
        # Get input/output dimensions
        self.state_dim = n_states
        self.action_dim = n_actions

        # Hidden layer dimension
        self.hidden_dim = n_hidden

        # Learning rate
        self.learn_rate = 1e-5

        # Create neural network for Gaussian policy (Actor)
        self.actor_network = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.action_dim),
            nn.Softmax(dim=-1)
        )

        # Create neural network for critic
        self.critic_network = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        )

        # Constant std deviations
        self.std = [1, 1]
        self.std_t = torch.tensor(self.std)

        # optimizers
        self.actor_optimizer = optim.Adam(self.actor_network.parameters(), lr=self.learn_rate)
        self.critic_optimizer = optim.Adam(self.critic_network.parameters(), lr=self.learn_rate)

    def get_action_dist(self, state):
        # Convert the state into a tensor
        state_t = torch.from_numpy(state)

        # Get the mean action
        mean_action = self.actor_network(state_t.float())

        # Sample from the normal distribution for the actions
        action_dist = Normal(mean_action, self.std_t)
        action_t = action_dist.sample()

        return action_dist

    def get_action(self, state):
        # Get the action distribution and sample from it
        dist = self.get_action_dist(state)
        action_t = dist.sample()

        # Compute the log probability
        log_prob = dist.log_prob(action_t).unsqueeze(0)

        # Return the numpy action and the log probability
        return (action_t.detach().numpy(), log_prob)

    def get_value(self, state):
        # Convert the state into a tensor
        state_t = torch.from_numpy(state)

        # Get the critic value
        value_t = self.critic_network(state_t.float())

        return value_t

    def update(self, rewards, log_probs, values):
        # Convert everything into tensors
        rewards_t = torch.tensor(rewards)
        values_t = torch.cat(values)
        log_probs_t = torch.cat(log_probs)

        # Transpose the log probabilities
        log_probs_t = torch.transpose(log_probs_t, 0, 1)

        # Compute Advantage
        advantage_t = rewards_t - values_t

        # Loss functions
        actor_loss_t = -1 * (log_probs_t * advantage_t.detach()).mean()
        critic_loss_t = advantage_t.pow(2).mean()

        # Set up optimization
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()

        # Do the backpropagation
        actor_loss_t.backward()
        critic_loss_t.backward()

        # Do the optimization step
        self.actor_optimizer.step()
        self.critic_optimizer.step()
