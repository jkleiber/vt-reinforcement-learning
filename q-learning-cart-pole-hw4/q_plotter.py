
import pickle
import matplotlib.pyplot as plt
import numpy as np


def plot_data():
    # import pickle data
    with open("Q_results.pkl", "rb") as f:
        Q_dict = pickle.load(f)

    # Keep track of figures
    i = 1

    # Get the settings
    n_episodes = Q_dict['settings']['n_episodes']

    # Get the data
    data = Q_dict['sims']

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
    plt.title(f"Q Function Rewards")

    # Save the figure as PNG
    plt.savefig(f'Q_fun_results.png', bbox_inches='tight')
    
    # Iterate figure tracker
    i = i + 1
        
    print(Q_dict['traj'][n_episodes - 1])

    # show the results
    # plt.show()

    print(Q_dict['left_theta'])
    # print(Q_dict['right_theta'])



if __name__ == "__main__":
    plot_data()