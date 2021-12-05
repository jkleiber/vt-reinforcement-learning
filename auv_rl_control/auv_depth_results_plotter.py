
import pickle
import matplotlib.pyplot as plt
import numpy as np


# Depth control action map
angles = np.radians(np.linspace(-10, 10, num=41))
action_map = angles
print(np.size(angles))

def plot_depth_data(results, ep = 964):
    # Find number of episodes
    n_episodes = len(results)

    # Convert the results into vectors
    rewards = [np.sum(result["rewards"]) for result in results]
    actions = [result["actions"] for result in results]
    states = [result["states"] for result in results]

    # Create an episode list
    eps = np.linspace(1, n_episodes, num=n_episodes)

    # Plot the reward data
    plot1 = plt.figure(1)
    plt.plot(eps, rewards)

    # Labels
    plt.xlabel("Episode #")
    plt.ylabel("Simulation Reward")
    plt.suptitle(f"Reward")

    # Save the figure as PNG
    plt.savefig(f'depth_reward.png', bbox_inches='tight')


    # Plot the last figure trajectory data
    X = np.squeeze(states[ep-1])
    n_steps = len(X)
    steps = np.linspace(1,n_steps,num=n_steps)

    plot2 = plt.figure(2)
    plt.plot(steps, X)
    plt.legend(("Pitch Error", "Pitch Rate", "Depth Error"))
    plt.suptitle("Trajectory")
    plt.title(f"Episode: {ep}")
    plt.xlabel("Time")

    # Save the figure as PNG
    plt.savefig(f'depth_traj.png', bbox_inches='tight')


    # Control plots
    ctrl = np.degrees(action_map[actions[ep - 1]])
    plot3 = plt.figure(3)
    plt.plot(steps,ctrl)
    plt.suptitle("Control")
    plt.title(f"Episode: {ep}")
    plt.xlabel("Time")
    plt.legend("Elevator")

    plt.savefig(f'depth_control.png', bbox_inches='tight')


    print("Plotting finished!")




if __name__ == "__main__":
    # Pick results file
    results_file = "unsafe_train_dqn_depth_results.pkl"

    # import pickle data
    with open(results_file, "rb") as f:
        results = pickle.load(f)

    plot_depth_data(results)