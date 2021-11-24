
import gym

import pickle
import numpy as np

# Plotting
from auv_results_plotter import plot_data

# Policies
from policy.gaussian_policy import GaussianPolicy

# The AUV environment
env = gym.make("gym_auv:AUVControl-v0")

# Training parameters
n_episodes = 500
n_train_steps = 50


# Algorithms
gaussian_policy = GaussianPolicy()

def simulate(policy, n_steps):
    # Reset the environment
    state = env.reset()

    # Track trajectory
    states = []
    actions = []
    rewards = []
    log_probs = []
    values = []

    for i in range(n_steps):
        # Determine action
        (u, log_prob) = policy.get_action(state)

        # Get the value from the critic
        value = policy.get_value(state)

        # Get the next state, reward and termination flag
        next_state, reward, done, info = env.step(u)

        # Append to the trajectory
        states.append(state)
        actions.append(u)
        rewards.append(reward)
        log_probs.append(log_prob)
        values.append(value)

        # End early if the environment says to
        if done:
            print(f"Done early: {i}")
            break

        # Update the state
        state = next_state

    trajectory = {
        "states": states,
        "actions": actions,
        "rewards": rewards,
        "log_probs": log_probs,
        "values": values
    }

    # Return trajectory
    return trajectory

def train(policy, n_episodes, n_steps):
    results = []

    # Simulate N times
    for ep in range(n_episodes):
        trajectory = simulate(policy, n_steps)

        # save the trajectory
        results.append(trajectory)

        # Unpack the trajectory
        states = trajectory["states"]
        actions = trajectory["actions"]
        rewards = trajectory["rewards"]
        log_probs = trajectory["log_probs"]
        values = trajectory["values"]

        # Print the total reward
        print(f"Episode #{ep+1} - Reward: {np.sum(rewards)}")

        # update the policy
        policy.update(rewards, log_probs, values)

    # Return the results
    return results


if __name__ == "__main__":
    # Choose the policy to use
    train_policy = gaussian_policy

    # Train the algorithm
    train_results = train(gaussian_policy, n_episodes, n_train_steps)

    # Export results to pickle
    with open("unsafe_train_results.pkl", "wb") as f:
        pickle.dump(train_results, f)

    # Plot the training data
    plot_data(train_results)