
import gym
import itertools
import numpy as np
import pickle

from math import pi

from pg_plotter import plot_data

# Gym environment
env = None

# Action definitions
LEFT = 0
RIGHT = 1

# Environment parameters
n_states = 4
n_actions = 2

# TD Learning parameters
alpha = 1e-5 # Step size
gamma = 0.9 # Gamma

# Simulation settings
n_episodes = 200

# LFA weights
left_theta = None
right_theta = None

# Random generator
gen = np.random.default_rng()

# Fourier basis parameters
n_f_basis = 7
n_basis_fun = (n_f_basis + 1)**n_states



# Results
PG_dict = {}


def init():
    global env, left_theta, right_theta, PG_dict

    # Make the environment
    env = gym.make('CartPole-v1')

    # Save the settings
    PG_dict['settings'] = {
        'n_episodes': n_episodes, 
        'f_order': n_f_basis, 
        'n_basis_fun': n_basis_fun
        }

    # Simulation dictionary
    PG_dict['sims'] = []
    PG_dict['traj'] = []
    PG_dict['theta'] = None

    # Initialize the PG function
    left_theta = np.zeros(n_basis_fun)
    right_theta = np.zeros(n_basis_fun)


def F_basis(state):
    # Multivariate Fourier basis for states
    c = np.array( list( itertools.product(range(n_f_basis+1), repeat=n_states) ), dtype=np.int32)

    # Determine feature vector
    in_prod = np.dot(c, state)

    # Fourier basis
    phi = np.cos(in_prod * pi)

    # Return basis vector
    return phi

def policy_gradient_lfa(s,a):
    global left_theta, right_theta

    # Choose coeffs based on action
    coeffs = None
    if a == 0:
        coeffs = left_theta
    else:
        coeffs = right_theta

    phi = F_basis(s)
    value = np.dot(phi, coeffs)

    # Return the LFA
    return value


def normalize_weights():
    global left_theta, right_theta

    if np.linalg.norm(right_theta,2) > 1:
        right_theta /= np.linalg.norm(right_theta,2)

    if np.linalg.norm(left_theta,2) > 1:
        left_theta /= np.linalg.norm(left_theta,2)


def softmax_probability(s,a):
    # Exponentials for each action's weights
    exp_left = np.exp(policy_gradient_lfa(s, LEFT))
    exp_right = np.exp(policy_gradient_lfa(s, RIGHT))
    exp_action = [exp_left, exp_right]
    exp_total = exp_left + exp_right

    # Probabilities for each action
    soft_prob = exp_action[a] / exp_total

    return soft_prob


def softmax_gradient(s, a):
    action_prob = softmax_probability(s,a)
    gradient = action_prob * (1 - action_prob)

    return gradient



def softmax_action_sample(s):
    global left_theta, right_theta

    p_left = softmax_probability(s, LEFT)

    # Decide which action to take
    a = gen.uniform(0,1)

    if a <= p_left:
        return LEFT
    
    return RIGHT


def simulate(ep_num, render=True):
    global env, gen, left_theta, right_theta

    # Restart the environment
    obs = env.reset()

    # Total reward
    trajectory = {
        "reward": [],
        "states": [],
        "actions": []
    }

    # Max values
    max_obs = np.array([4.8/2, 5, 0.418/2, 5])
    min_obs = np.array([-4.8/2, -5, -0.418/2, -5])
    diff = max_obs - min_obs

    # Run until the pole falls or we win
    done = False
    i = 0
    while not done:
        # Scale the observation to [0,1]
        obs -= min_obs
        obs /= diff

        # Add state to the trajectory
        trajectory["states"].append(obs)

        # Render the environment
        if render:
            env.render()

        # Exit if the weights have become nan
        if np.isnan(left_theta).any() or np.isnan(right_theta).any():
            print(f"NAN: left: {np.isnan(left_theta).any()} or right: {np.isnan(right_theta).any()}")
            exit(-1)

        # Determine Policy gradient action using softmax
        action = softmax_action_sample(obs)

        # Step in the environment
        new_obs, reward, done, info = env.step(action)

        # Add action to the trajectory
        trajectory["actions"].append(action)

        # Add to the reward trajectory
        trajectory["reward"].append(reward)

        # If the program is done, end it
        if done:
            break

        # Update the observation for the next step
        obs = new_obs

        # Update step count
        i += 1
    # end simulation steps

    # Return the total reward for plotting
    return trajectory


def run_main():
    global PG_dict, right_theta, left_theta

    # Initialize the gym environment
    init()

    # Run some amount of episodes
    for i in range(n_episodes):
        # Get the reward from the simulation
        trajectory = simulate(i, render = False)

        # Save trajectory results
        PG_dict['traj'].append(trajectory)
        PG_dict['left_theta'] = left_theta
        PG_dict['right_theta'] = right_theta

        # Save reward results
        reward = sum(trajectory["reward"])
        PG_dict["sims"].append(reward)

        print(f"Episode #{i+1} - Reward: {reward}")

        # Update the weights
        states = trajectory["states"]
        actions = trajectory["actions"]
        rewards = trajectory["reward"]
        T = len(states)

        for t in range(T):
            # Compute G
            G = 0
            for k in range(t+1, T):
                G += gamma**(k-t-1) * rewards[k]

            # update theta
            state_t = states[t]
            action_t = actions[t]
            phi = F_basis(state_t)
            update = (alpha / (1 - gamma)) * G * phi * softmax_gradient(state_t, action_t)
            # update = alpha * (gamma**t) * G * softmax_gradient(state_t, action_t)

            if action_t == LEFT:
                left_theta += update 
            else:
                right_theta += update

            # Normalize the weights
            normalize_weights()

        # Solution checking
        if i >= 100:
            reward_arr = np.array(PG_dict['sims'])
            reward_avg = np.average(reward_arr[(i-100):i])

            if reward_avg >= 195:
                print(f"Cart Pole Solved in {i+1} episodes.")
                PG_dict['settings']['n_episodes'] = i+1
                break
    
    # end episodes for-loop

    # Export results to pickle
    with open("PG_results.pkl", "wb") as f:
        pickle.dump(PG_dict, f)

    # Plotting tool
    plot_data()



# Run the main code.
if __name__ == "__main__":
    run_main()