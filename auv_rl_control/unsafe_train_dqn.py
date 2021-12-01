
import gym

import pickle
import numpy as np

# Plotting
from auv_results_plotter import plot_data

# Policies
from policy.dqn_policy import DQNPolicy

# The AUV environment
env = gym.make("gym_auv:AUVControl-v0")

# Training parameters
n_episodes = 500
n_train_steps = 200


# Action map is a meshgrid of values
angles = np.array([-10, -5, -2, 0, 2, 5, 10])
rudder, elevator = np.meshgrid(angles, angles)
rudder_flat = rudder.flatten()
elevator_flat = elevator.flatten()
action_map = np.stack((elevator_flat, rudder_flat))

# Algorithms
dqn_policy = DQNPolicy(action_map)

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
        u_idx = policy.get_action(state)
        u = action_map[:,u_idx]

        # Get the next state, reward and termination flag
        next_state, reward, done, info = env.step(u)

        # Append to the trajectory
        states.append(state)
        actions.append(u_idx)
        rewards.append(reward)

        # Add this experience to the replay buffer
        dqn_policy.add_experience(state, u_idx, reward, next_state, done)

        # Train the policy
        policy.update()

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

        # Reduce the epsilon value
        policy.reduce_epsilon()

    # Return the results
    return results


if __name__ == "__main__":
    # Choose the policy to use
    train_policy = dqn_policy

    # Train the algorithm
    train_results = train(dqn_policy, n_episodes, n_train_steps)

    # Export results to pickle
    with open("unsafe_train_dqn_results.pkl", "wb") as f:
        pickle.dump(train_results, f)

    # Save the model information
    train_policy.save_models()

    # Plot the training data
    plot_data(train_results)