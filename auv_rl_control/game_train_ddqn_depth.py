
import gym

import pickle
import numpy as np

# Plotting
from auv_depth_results_plotter import plot_depth_data

# Policies
from policy.ddqn_policy import DDQNPolicy

# The AUV environment
env = gym.make("gym_auv:AUVDepthControlGame-v0")

# Training parameters
n_episodes = 1000
n_train_steps = 200
n_target_update = 2500
n_policy_steps = 5

# Track steps taken before target update
target_step_track = 0

# Depth control action map for protagonist
angles = np.radians(np.linspace(-10, 10, num=41))
action_map = angles
# Adversary actions
ant_angles = np.radians(np.linspace(-0.45, 0.45, num=41))
ant_action_map = ant_angles

# Algorithms
pro_policy = DDQNPolicy(action_map, n_states=3, n_actions=np.size(angles), n_hidden=16, lr=1e-3, prefix="pro_game")
ant_policy = DDQNPolicy(ant_action_map, n_states=3, n_actions=np.size(ant_angles), n_hidden=16, lr=1e-3, prefix="ant_game")

def simulate(pro_policy, ant_policy, n_steps):
    global target_step_track

    # Reset the environment
    state = env.reset()

    # Track trajectory
    states = []
    actions = []
    rewards = []
    log_probs = []
    values = []

    pro_phase = True
    phase_track = 0

    for i in range(n_steps):
        # Determine action
        u_idx = pro_policy.get_action(state)
        v_idx = ant_policy.get_action(state)
        u = action_map[u_idx]
        v = ant_action_map[v_idx]

        ctrl = np.array([u,v])

        # Get the next state, reward and termination flag
        next_state, reward, done, info = env.step(ctrl)

        # Append to the trajectory
        states.append(state)
        actions.append(u_idx)
        rewards.append(reward)

        # Train the protagonist policy
        if pro_phase:
            # Add this experience to the replay buffer
            pro_policy.add_experience(state, u_idx, reward, next_state, done)

            # Train the protagonist policy
            pro_policy.update()
        # Otherwise train the antagonist policy
        else:
            # Add this experience to the replay buffer
            ant_policy.add_experience(state, v_idx, -1*reward, next_state, done)

            # Train the protagonist policy
            ant_policy.update()


        # End early if the environment says to
        if done:
            print(f"Done early: {i}")
            break

        # Update the state
        state = next_state

        # Update the target step tracking
        target_step_track += 1
        phase_track += 1

        # Update the phase
        if phase_track >= n_policy_steps:
            pro_phase = not pro_phase

        if target_step_track >= n_target_update:
            pro_policy.update_target()
            ant_policy.update_target()
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

def train(pro_policy, ant_policy, n_episodes, n_steps):
    results = []

    # Simulate N times
    for ep in range(n_episodes):
        trajectory = simulate(pro_policy, ant_policy, n_steps)

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
        pro_policy.reduce_epsilon(reduction=0.0015)
        ant_policy.reduce_epsilon(reduction=0.0015)

    # Return the results
    return results


if __name__ == "__main__":
    # Train the algorithm
    train_results = train(pro_policy, ant_policy, n_episodes, n_train_steps)

    # Export results to pickle
    with open("game_train_ddqn_depth_results.pkl", "wb") as f:
        pickle.dump(train_results, f)

    # Save the model information
    pro_policy.save_models()
    ant_policy.save_models()

    # Plot the training data
    plot_depth_data(train_results)