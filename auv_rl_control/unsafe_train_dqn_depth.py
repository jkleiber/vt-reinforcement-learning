
import gym

import pickle
import numpy as np

# Plotting
from auv_depth_results_plotter import plot_depth_data

# Policies
from policy.ddqn_policy import DDQNPolicy
from policy.ddqn_cbf import DDQNPolicyCBF

# The AUV environment
env = gym.make("gym_auv:AUVDepthControl-v0")

# Training parameters
n_episodes = 1000
n_train_steps = 200
n_target_update = 2500

# Track steps taken before target update
target_step_track = 0

# Depth control action map
angles = np.radians(np.linspace(-10, 10, num=41))
action_map = angles

# Algorithms
dqn_policy = DDQNPolicyCBF(action_map, n_states=3, n_actions=np.size(angles), n_hidden=32)

def simulate(policy, n_steps):
    global target_step_track

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
        u = action_map[u_idx]

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

        # Update the target step tracking
        target_step_track += 1

        if target_step_track >= n_target_update:
            policy.update_target()
            target_step_track = 0

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
        policy.reduce_epsilon(reduction=0.0015)

    # Return the results
    return results


if __name__ == "__main__":
    # Choose the policy to use
    train_policy = dqn_policy

    # Train the algorithm
    train_results = train(dqn_policy, n_episodes, n_train_steps)

    # Export results to pickle
    with open("unsafe_train_dqn_depth_results.pkl", "wb") as f:
        pickle.dump(train_results, f)

    # Save the model information
    train_policy.save_models()

    # Plot the training data
    plot_depth_data(train_results)