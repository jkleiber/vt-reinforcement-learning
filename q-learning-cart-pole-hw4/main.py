
import gym
import itertools
import numpy as np
import pickle

from math import pi

from q_plotter import plot_data

# Gym environment
env = None

# Action definitions
LEFT = 0
RIGHT = 1

# Environment parameters
n_states = 4
n_actions = 2

# TD Learning parameters
alpha = 0.001 # Step size
epsilon = 0.01 # Chance of random action
gamma = 0.9 # Gamma

# Simulation settings
n_episodes = 1000

# Q function (tabular)
# theta = None
left_theta = None
right_theta = None

# Random generator
gen = np.random.default_rng()

# Fourier basis parameters
n_f_basis = 3
n_basis_fun = (n_f_basis + 1)**n_states



# Results
Q_dict = {}


def init():
    global env, left_theta, right_theta, Q_dict

    # Make the environment
    env = gym.make('CartPole-v1')

    # Save the settings
    Q_dict['settings'] = {
        'n_episodes': n_episodes, 
        'f_order': n_f_basis, 
        'n_basis_fun': n_basis_fun
        }

    # Simulation dictionary
    Q_dict['sims'] = []
    Q_dict['traj'] = {}
    Q_dict['theta'] = None

    # Initialize the Q function
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

def Q_function(s,a):
    global left_theta, right_theta

    # Choose coeffs based on action
    coeffs = None
    if a == 0:
        coeffs = left_theta
    else:
        coeffs = right_theta

    phi = F_basis(s)
    value = np.dot(phi, coeffs)

    # Return the Q-value
    return value


def normalize_weights():
    global left_theta, right_theta

    if np.linalg.norm(right_theta,2) > 1:
        right_theta /= np.linalg.norm(right_theta,2)

    if np.linalg.norm(left_theta,2) > 1:
        left_theta /= np.linalg.norm(left_theta,2)


def simulate(ep_num, render=True):
    global env, Q_fun, gen, left_theta, right_theta

    # Restart the environment
    obs = env.reset()
    Q_dict['traj'][ep_num].append(obs)

    # Total reward
    total_reward = 0

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

        # Render the environment
        if render:
            env.render()

        # Random action
        rand_action = env.action_space.sample()

        # Exit if the weights have become nan
        if np.isnan(left_theta).any() or np.isnan(right_theta).any():
            print(f"NAN: left: {np.isnan(left_theta).any()} or right: {np.isnan(right_theta).any()}")
            exit(-1)

        # Q function calculation
        left_val = Q_function(obs, LEFT)
        right_val = Q_function(obs, RIGHT)

        # Determine Q action
        Q_action = LEFT
        if right_val > left_val:
            Q_action = RIGHT

        # Decide which action to take
        a = gen.uniform(0,1)
        if ep_num < 1:
            action = rand_action
        elif a > epsilon:
            action = Q_action
        else:
            action = rand_action

        # Step in the environment
        new_obs, reward, done, info = env.step(action)

        # Save a new trajectory point
        Q_dict['traj'][ep_num].append(new_obs)

        # Scale new observation
        new_obs_s = new_obs - min_obs
        new_obs_s /= diff

        # Find best action for next state
        new_left_val = Q_function(new_obs_s, LEFT)
        new_right_val = Q_function(new_obs_s, RIGHT)
        best_val = new_left_val if new_left_val > new_right_val else new_right_val

        # Q function update
        update = alpha * (reward + gamma*best_val - Q_function(obs, action)) * F_basis(obs)
        
        if action == LEFT:
            left_theta += update
        else:
            right_theta += update

        # Add to the total reward
        total_reward += reward

        # If the program is done, end it
        if done:
            break

        # Update the observation for the next step
        obs = new_obs

        # Normalize the weights
        normalize_weights()

        # Update step count
        i += 1
    # end simulation steps

    # Return the total reward for plotting
    return total_reward


def run_main():
    global right_theta, left_theta

    # Initialize the gym environment
    init()

    # Run some amount of episodes
    for i in range(n_episodes):
        print(f"Episode #{i+1}")

        # Save the trajectories
        Q_dict['traj'][i] = []

        # Get the reward from the simulation
        reward = simulate(i, render = False)

        Q_dict['sims'].append(reward)
        Q_dict['left_theta'] = left_theta
        Q_dict['right_theta'] = right_theta

        if i >= 100:
            reward_arr = np.array(Q_dict['sims'])
            reward_avg = np.average(reward_arr[(i - 100):i])

            if reward_avg >= 195:
                print(f"Cart Pole Solved in {i+1} episodes.")
                Q_dict['settings']['n_episodes'] = i+1
                break
    
    # end episodes for-loop

    # Export Q results to pickle
    with open("Q_results.pkl", "wb") as f:
        pickle.dump(Q_dict, f)

    # Plotting tool
    plot_data()



# Run the main code.
if __name__ == "__main__":
    run_main()