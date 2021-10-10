
import pickle
import matplotlib.pyplot as plt
import numpy as np


def plot_data():
    # import pickle data
    with open("neu_results.pkl", "rb") as f:
        neu_dict = pickle.load(f)

    # Keep track of figures
    i = 1

    # Get the settings
    n_episodes = neu_dict['settings']['n_episodes']

    # Plot the data for each NEU key
    for key in neu_dict['simulations']:
        # Get the data
        data = neu_dict['simulations'][key]

        # Convert the data to numpy array
        np_data = np.array(data)

        # Get number of episode data points
        n_eps_data = len(data)

        # Create an episode list
        eps = np.linspace(1, n_episodes, num=n_eps_data)

        # Plot the data
        plot1 = plt.figure(i)
        plt.plot(eps, np_data)

        # Labels
        plt.xlabel("Episode #")
        plt.ylabel("NEU Value")
        plt.title(f"λ = {key}")

        # Save the figure as PNG
        plt.savefig(f'lambda_{key}.png', bbox_inches='tight')
        
        # Iterate figure tracker
        i = i + 1
        
    # end plot for-loop

    # show the results
    # plt.show()



if __name__ == "__main__":
    plot_data()