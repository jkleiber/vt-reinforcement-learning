
import gym

import pickle
import numpy as np

# Plotting
from auv_depth_results_plotter import plot_depth_data, plot_test_data

# Policies
from policy.dqn_policy import DQNPolicy
from policy.ddqn_policy import DDQNPolicy
from policy.ddqn_cbf import DDQNPolicyCBF

# The AUV environment
env = gym.make("gym_auv:AUVDepthControl-v0")

# Training parameters
n_episodes = 100
n_test_steps = 200

# Depth control action map
angles = np.radians(np.linspace(-10, 10, num=41))
action_map = angles

# Algorithms
unsafe_policy = DDQNPolicy(action_map, n_states=3, n_actions=np.size(angles), n_hidden=16, lr=1e-3, eps = 0, prefix="base")
robust_policy = DDQNPolicy(action_map, n_states=3, n_actions=np.size(angles), n_hidden=16, lr=1e-3, eps = 0, prefix="pro_game")

# Averages
averages = []

def simulate(policy, n_steps):
    # Reset the environment
    state = env.reset()

    # Track trajectory
    states = []
    actions = []
    rewards = []

    for i in range(n_steps):
        # Determine action
        u_idx = policy.get_action(state)
        u = action_map[u_idx]

        # Get the next state, reward and termination flag
        next_state, reward, done, info = env.step(u)

        # Append to the trajectory
        states.append(state)
        actions.append(u_idx)
        rewards.append(reward)

        # End early if the environment says to
        if done:
            # print(f"Done early: {i}")
            break

        # Update the state
        state = next_state

    trajectory = {
        "states": states,
        "actions": actions,
        "rewards": rewards
    }

    # Return trajectory
    return trajectory

def test(policy, policy_name, n_steps):
    global averages
    results = []
    total_reward = 0

    for i in range(n_episodes):
        trajectory = simulate(policy, n_steps)

        # save the trajectory
        results.append(trajectory)

        # Unpack the trajectory
        states = trajectory["states"]
        actions = trajectory["actions"]
        rewards = trajectory["rewards"]

        # Update total reward
        R = np.sum(rewards)
        total_reward += R

        print(f"{policy_name} episode {i} test reward: {R}")

    # Print the total reward
    avg_reward = total_reward / n_episodes
    averages.append(avg_reward)
    print(f"{policy_name} average reward: {avg_reward}")

    # Return the results
    return results


if __name__ == "__main__":
    # Policies to test
    policies = [unsafe_policy, robust_policy]
    policy_names = ["unsafe_policy", "robust_policy"]

    # Test each policy
    for i in range(len(policies)):
        # Choose the policy to use
        policy = policies[i]

        # Load the policy models
        policy.load_models()

        # Train the algorithm
        test_results = None
        test_results = test(policy, policy_names[i], n_test_steps)

        # Plot the test data
        plot_test_data(test_results, policy_names[i])

    for i in range(len(policies)):
        print(f"{policy_names[i]} average reward: {averages[i]}")