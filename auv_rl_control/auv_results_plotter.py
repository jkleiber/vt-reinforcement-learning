
import pickle
import matplotlib.pyplot as plt
import numpy as np


angles = np.array([-10, -5, -2, 0, 2, 5, 10])
rudder, elevator = np.meshgrid(angles, angles)
rudder_flat = rudder.flatten()
elevator_flat = elevator.flatten()
action_map = np.stack((elevator_flat, rudder_flat))

def plot_data(results):
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
    plt.title(f"Reward")

    # Save the figure as PNG
    plt.savefig(f'reward.png', bbox_inches='tight')


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
    plt.savefig(f'traj.png', bbox_inches='tight')


    # Control plots
    print(action_map[:,actions[n_episodes - 1]])
    ctrl = np.transpose(action_map[:,actions[n_episodes - 1]])
    plot3 = plt.figure(3)
    plt.plot(steps,ctrl)
    plt.title("Control")
    plt.xlabel("Time")
    plt.legend(("Elevator", "Rudder"))

    plt.savefig(f'control.png', bbox_inches='tight')




if __name__ == "__main__":
    # Pick results file
    results_file = "unsafe_train_dqn_results.pkl"

    # import pickle data
    with open(results_file, "rb") as f:
        results = pickle.load(f)

    plot_data(results)