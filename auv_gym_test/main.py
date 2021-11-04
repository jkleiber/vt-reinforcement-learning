
from math import radians
import gym

import numpy as np

env = gym.make("gym_auv:AUVControl-v0")

print(env.reset())
print("beginning simulation")

for i in range(150):
    u = np.array([radians(-2), 0], dtype=np.float32)

    obs, reward, done, info = env.step(u)

    print(obs)

    if done:
        print(f"Done early: {i}")
        break