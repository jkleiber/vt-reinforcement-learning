
import pickle
import matplotlib.pyplot as plt
import numpy as np


def plot_data():
    # import pickle data
    with open("PG_results.pkl", "rb") as f:
        PG_dict = pickle.load(f)

    # Keep track of figures
    i = 1

    # Get the settings
    n_episodes = PG_dict['settings']['n_episodes']

    # Get the data
    data = PG_dict['sims']

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
    plt.ylabel("Simulation Reward")
    plt.title(f"Policy Gradient Rewards")

    # Save the figure as PNG
    plt.savefig(f'PG_results.png', bbox_inches='tight')
    
    # Iterate figure tracker
    i = i + 1
        
    print(PG_dict['traj'][n_episodes - 1])

    # show the results
    # plt.show()

    print(PG_dict['left_theta'])
    # print(PG_dict['right_theta'])



if __name__ == "__main__":
    plot_data()