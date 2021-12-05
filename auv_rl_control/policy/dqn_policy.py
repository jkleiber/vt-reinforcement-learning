from buffer.replay_buffer import ReplayBuffer
from policy.policy import Policy

from math import radians
import numpy as np

# PyTorch
import torch
from torch import nn
from torch import optim
from torch.distributions import Normal

class DQNPolicy(Policy):

    BATCH_SIZE = 32

    def __init__(self, action_map, n_states = 3, n_actions = 41, n_hidden = 32, lr = 1e-4, gamma = 0.99):
        # Map action index to action value
        self.action_map = action_map
        
        # Get input/output dimensions
        self.state_dim = n_states
        self.action_dim = n_actions

        # Hidden layer dimension
        self.hidden_dim = n_hidden

        # Learning rate
        self.learn_rate = lr

        # Discount factor
        self.gamma = gamma

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
        self.replay_buffer = ReplayBuffer(max_size=100000)

        # number of training steps
        self.n_train_steps = 0


    def get_action(self, state):
        # Epsilon greedy
        a = torch.rand(1)

        best_val_idx = 0
        if a < self.epsilon:
            random_value_fn = torch.rand(self.action_dim)
            best_val_idx = torch.argmax(random_value_fn).int()
        else:
            # Convert the state into a tensor
            state_t = torch.from_numpy(state).float()

            # Estimate q function
            q_function = self.main_network(state_t)
            best_val_idx = torch.argmax(q_function).int()

        return best_val_idx

    def update(self):
        # Don't train if the batch size is bigger than our memory
        if self.BATCH_SIZE > self.replay_buffer.size():
            return
        
        # Sample a batch
        batch = self.replay_buffer.sample_experiences(self.BATCH_SIZE)

        # Convert batch elements to a ndarray
        states_np = np.array(batch[0])
        actions_np = np.array(batch[1])
        rewards_np = np.array(batch[2])
        next_states_np = np.array(batch[3])
        dones_np = np.array(batch[4])

        # Convert batch elements into tensors
        states_t = torch.tensor(states_np).float()
        actions_t = torch.tensor(actions_np).float()
        rewards_t = torch.tensor(rewards_np).float()
        next_states_t = torch.tensor(next_states_np).float()
        dones_t = torch.tensor(dones_np).float()

        # Find the target network's Q value prediction
        target_q_values = self.target_network(next_states_t)
        max_target_q = torch.max(target_q_values)
        new_q = rewards_t + (1 - dones_t) * self.gamma * max_target_q

        # Find the Q values from the main network
        old_q = self.main_network(states_t)
        action_onehot = nn.functional.one_hot(actions_t.long(), num_classes = self.action_dim)
        main_q = torch.sum(action_onehot * old_q, dim = -1)

        # Compute the MSE
        mse_loss = nn.MSELoss()
        loss = mse_loss(new_q, main_q)

        # Optimize
        self.main_optimizer.zero_grad()
        loss.backward()
        self.main_optimizer.step()

    def update_target(self):
        self.target_network.load_state_dict(self.main_network.state_dict())            

    def add_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.append(state, action, reward, next_state, done)

    def reduce_epsilon(self, reduction = 0.01):
        if self.epsilon > 0:
            self.epsilon -= reduction

    def save_models(self):
        torch.save(self.main_network.state_dict(), "main_net.pt")
        torch.save(self.target_network.state_dict(), "target_net.pt")

    def load_models(self):
        self.main_network.load_state_dict(torch.load("main_net.pt"))
        self.target_network.load_state_dict(torch.load("target_net.pt"))

        self.main_network.eval()
        self.target_network.eval()
