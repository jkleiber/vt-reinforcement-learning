
import gym
import numpy as np
import pickle

from numpy.lib.function_base import gradient

from pg_plotter import plot_data

# Gym environment
env = None

# TD Learning parameters
alpha = 1e-4 # Step size
gamma = 0.9  # Gamma

# Simulation settings
n_episodes = 1000
n_steps = 200

# States and actions
n_states = 48
n_actions = 4

# Q function (tabular)
Theta = None

# Random generator
gen = np.random.default_rng(1234)


# Results
PG_dict = {}


def init():
    global env, Theta, PG_dict

    # Make the environment
    env = gym.make('gym_cliffwalking:cliffwalking-v0')

    # Save the settings
    PG_dict['settings'] = {'n_episodes': n_episodes}

    # Simulation dictionary
    PG_dict['sims'] = []
    PG_dict['traj'] = []
    PG_dict['num_success'] = 0

    # Initialize the Q function
    # 48 states and 4 actions
    Theta = np.zeros((n_states, n_actions))

    # Visitation heatmap
    PG_dict['heatmap'] = np.zeros((n_states))



def softmax_probability(s):
    global Theta

    # Exponentials for each action's weights
    exp_list = []
    for i in range(n_actions):
        action_exp = np.exp(Theta[s, i])
        exp_list.append(action_exp)

    # Total exp sum
    exp_total = sum(exp_list)

    # Compute probabilities for each action
    action_probs = []
    for i in range(n_actions):
        soft_prob = exp_list[i] / exp_total
        action_probs.append(soft_prob)

    return action_probs


def softmax_gradient(s, a):
    # Get action probabilities
    action_prob = softmax_probability(s)

    # Compute the gradients
    gradient = []
    for i in range(n_actions):
        grad = action_prob[i] * (1 - action_prob[i])
        gradient.append(grad)

    return np.array(gradient)



def softmax_action_sample(s):
    # Get action probabilities
    action_probs = softmax_probability(s)

    # Sample from the possible actions
    action = gen.choice(range(4), size=1, p=action_probs)[0]

    return action



def simulate(ep_num):
    global env, Theta, gen

    # Restart the environment
    obs = env.reset()

    # Total reward
    trajectory = {
        "reward": [],
        "states": [],
        "actions": []
    }

    for i in range(n_steps):
        # Q function action
        action = softmax_action_sample(obs)

        # Step in the environment
        new_obs, reward, done, info = env.step(action)

        # Append to the trajectory
        trajectory["states"].append(obs)
        trajectory["actions"].append(action)
        trajectory["reward"].append(reward)

        # Add a visit to the heatmap
        PG_dict["heatmap"][obs] += 1

        # If the program is done, end it
        if done:
            PG_dict['num_success'] += 1
            break

        # Update the observation for the next step
        obs = new_obs
    # end simulation steps

    # Return the total reward for plotting
    return trajectory


def run_main():
    # Initialize the gym environment
    init()

    # Run some amount of episodes
    for i in range(n_episodes):
        
        # Get the reward from the simulation
        trajectory = simulate(i)

        # Save trajectory results
        PG_dict['traj'].append(trajectory)

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
            
            gradient =  softmax_gradient(state_t, action_t)
            update = alpha * gamma**t * G

            # Apply gradient to each parameter for state s_t
            for a in range(n_actions):
                if a == action_t:
                    Theta[state_t, action_t] += update * gradient[action_t]
                else:
                    Theta[state_t, a] += update * (1 - gradient[a])
        
        # Add table to PG data dictionary
        PG_dict["value_table"] = Theta
    
    # end episodes for-loop

    # Export results to pickle
    with open("PG_results.pkl", "wb") as f:
        pickle.dump(PG_dict, f)

    # Plotting tool
    plot_data()



# Run the main code.
if __name__ == "__main__":
    run_main()