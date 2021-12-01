
import numpy as np
from collections import deque

class ReplayBuffer():

    def __init__(self, max_size = 10000, seed = 1234):
        self.state_buf = deque(maxlen=max_size)
        self.action_buf = deque(maxlen=max_size)
        self.reward_buf = deque(maxlen=max_size)
        self.next_state_buf = deque(maxlen=max_size)
        self.done_buf = deque(maxlen=max_size)

        # Sample generator
        self.gen = np.random.default_rng(seed)


    def append(self,state, action, reward, next_state, done):
        self.state_buf.append(state)
        self.action_buf.append(action)
        self.reward_buf.append(reward)
        self.next_state_buf.append(next_state)
        self.done_buf.append(done)

    def sample_experiences(self, n_samples):
        buf_size = len(self.state_buf)

        idxs = self.gen.integers(0, buf_size, size=(n_samples,))

        state_list = [self.state_buf[idx] for idx in idxs]
        action_list = [self.action_buf[idx] for idx in idxs]
        reward_list = [self.reward_buf[idx] for idx in idxs]
        next_state_list = [self.next_state_buf[idx] for idx in idxs]
        done_list = [self.done_buf[idx] for idx in idxs]

        return (state_list, action_list, reward_list, next_state_list, done_list)

    def size(self):
        return len(self.state_buf)