
import gym
import numpy as np
import pickle

from td_plotter import plot_data

# Gym environment
env = None

# Values of lambda in TD(lambda) to test
lambdas = [0, 0.3, 0.5, 0.7, 1]

# TD Learning parameters
alpha = 0.1 # Step size
gamma = 0.9 # Gamma

# Simulation settings
n_episodes = 1000
n_steps = 200
n_avg_neu = 10  # Number of episodes to average for the NEU values


# Value function (tabular)
V_fun = []


# Results
neu_dict = {}


def init_v():
    for i in range(48):
        V_fun.append(0)


def init():
    global env, V_fun, neu_dict

    # Make the environment
    env = gym.make('gym_cliffwalking:cliffwalking-v0')

    # Save the settings
    neu_dict['settings'] = {'n_episodes': n_episodes, 'n_avg_neu': n_avg_neu}

    # Simulation dictionary
    neu_dict['simulations'] = {}

    # Initialize the value function
    init_v()


def simulate(L):
    global env, V_fun

    trace_vector = np.zeros(48) # Represent all trace states as 0 initially
    neu = 0

    obs = env.reset()

    for i in range(n_steps):
        # Take a random action
        action = env.action_space.sample()
        new_obs, reward, term, info = env.step(action)

        # Compute temporal difference
        d = reward + gamma*V_fun[new_obs] - V_fun[obs]

        # Compute current trace vector element
        z = L*gamma*trace_vector[obs] + 1

        # update the trace vector and value function at each state
        for j in range(48):
            if j != obs:
                trace_vector[j] = L*gamma*trace_vector[j]
            else:
                trace_vector[j] = z

            # Update the value function at state j
            V_fun[j] = V_fun[j] + alpha*d*trace_vector[j]
        # end update

        # Compute NEU for this state
        neu += (1/20)*(d * z)**2

        # If the simulation is over for some reason, end the loop
        if term:
            break

        # Update the observation for the next step of TD(lambda)
        obs = new_obs
    # end simulation steps

    # Return the NEU for plotting
    return neu


def run_main():
    # Initialize the gym environment
    init()

    # Simulate for each test value of lambda
    for L in lambdas:
        # For this value of lambda, create a list of NEUs so they can be plotted later
        neu_dict['simulations'][L] = []

        # Set the NEU sum tracker to 0
        neu_sum = 0

        # Initialize the value function
        init_v()

        # Run some amount of episodes
        for i in range(n_episodes):
            # Get the NEU from the simulation
            neu_value = simulate(L)

            # Add it to the running sum
            neu_sum += neu_value

            # Average the NEU values at a given frequency
            if (i+1) % n_avg_neu == 0:
                neu_avg = neu_sum / n_avg_neu
                neu_dict['simulations'][L].append(neu_avg)

                # Reset NEU sum
                neu_sum = 0
        
        # end episodes for-loop
    # end lambda for-loop

    # Export NEU results to pickle
    with open("neu_results.pkl", "wb") as f:
        pickle.dump(neu_dict, f)

    # Plotting tool
    plot_data()



# Run the main code.
if __name__ == "__main__":
    run_main()