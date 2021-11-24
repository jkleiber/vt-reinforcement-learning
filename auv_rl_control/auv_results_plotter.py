
import pickle
import matplotlib.pyplot as plt
import numpy as np


def plot_data(results):
    # Find number of episodes
    n_episodes = len(results)

    # Convert the results into vectors
    rewards = [np.sum(result["rewards"]) for result in results]
    states = [result["states"] for result in results]

    # Create an episode list
    eps = np.linspace(1, n_episodes, num=n_episodes)

    # Plot the reward data
    plot1 = plt.figure(1)
    plt.plot(eps, rewards)

    # Labels
    plt.xlabel("Episode #")
    plt.ylabel("Simulation Reward")
    plt.title(f"Actor-Critic Rewards")

    # Save the figure as PNG
    plt.savefig(f'AC_Reward.png', bbox_inches='tight')


    # Plot the last figure trajectory data
    X = np.squeeze(states[n_episodes-1])
    n_steps = len(X)
    steps = np.linspace(1,n_steps,num=n_steps)

    plot2 = plt.figure(2)
    plt.plot(steps, X)
    plt.legend(("Pitch Error", "Pitch Rate", "Yaw Error", "Yaw Rate", "Depth Error"))
    plt.title("Trajectory")
    plt.xlabel("Time")

    # Save the figure as PNG
    plt.savefig(f'AC_Traj.png', bbox_inches='tight')




if __name__ == "__main__":
    # Pick results file
    results_file = "unsafe_train_results.pkl"

    # import pickle data
    with open(results_file, "rb") as f:
        results = pickle.load(f)

    plot_data(results)