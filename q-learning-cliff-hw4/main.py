
import gym
import numpy as np
import pickle

from q_plotter import plot_data

# Gym environment
env = None

# TD Learning parameters
alpha = 0.1 # Step size
gamma = 0.9 # Gamma

# Simulation settings
n_episodes = 1000
n_steps = 50

# Q function (tabular)
Q_fun = None

# Random generator
gen = np.random.default_rng(69)


# Results
Q_dict = {}


def init():
    global env, Q_fun, Q_dict

    # Make the environment
    env = gym.make('gym_cliffwalking:cliffwalking-v0')

    # Save the settings
    Q_dict['settings'] = {'n_episodes': n_episodes}

    # Simulation dictionary
    Q_dict['sims'] = []
    Q_dict['traj'] = {}

    # Initialize the Q function
    # 48 states and 4 actions
    Q_fun = np.zeros((48, 4))


def simulate(ep_num):
    global env, Q_fun, gen

    # Restart the environment
    obs = env.reset()
    Q_dict['traj'][ep_num].append(obs)

    # Total reward
    total_reward = 0

    for i in range(n_steps):

        # Random action
        rand_action = env.action_space.sample()

        # Q function action
        Q_action = np.argmax(Q_fun[obs,:])

        # Decide which action to take
        alpha = gen.uniform(0,1)
        if alpha > 1 / (ep_num + 1):
            action = Q_action
        else:
            action = rand_action

        # Step in the environment
        new_obs, reward, done, info = env.step(action)

        # Save a new trajectory point
        Q_dict['traj'][ep_num].append(new_obs)

        # Q function update
        Q_fun[obs, action] = Q_fun[obs, action] + alpha*(reward + gamma*np.max(Q_fun[new_obs,:]) - Q_fun[obs,action])

        # Add to the total reward
        total_reward += reward

        # If the program is done, end it
        if done:
            break

        # Update the observation for the next step
        obs = new_obs
    # end simulation steps

    # Return the total reward for plotting
    return total_reward


def run_main():
    # Initialize the gym environment
    init()

    # Run some amount of episodes
    for i in range(n_episodes):
        # Save the trajectories
        Q_dict['traj'][i] = []

        # Get the reward from the simulation
        reward = simulate(i)

        Q_dict['sims'].append(reward)
    
    # end episodes for-loop

    # Export Q results to pickle
    with open("Q_results.pkl", "wb") as f:
        pickle.dump(Q_dict, f)

    # Plotting tool
    plot_data()



# Run the main code.
if __name__ == "__main__":
    run_main()