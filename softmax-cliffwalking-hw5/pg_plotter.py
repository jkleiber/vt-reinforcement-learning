
import pickle
import matplotlib.pyplot as plt
import numpy as np


def make_heatmap(X, Y, table, label = None):
    heatmap = plt.pcolormesh(X,Y, table, shading="auto", edgecolors="black")
    
    # heatmap = plt.pcolor(val_table)

    for y in range(table.shape[0]):
        for x in range(table.shape[1]):
            if label is None:
                plt.text(x + 0.5, y + 0.5, f'{table[y, x]:.2f} \n ({12*y + x})',
                        horizontalalignment='center',
                        verticalalignment='center',
                        )
            else:
                plt.text(x + 0.5, y + 0.5, f'{table[y, x]:.2f} \n ({12*y + x}) \n {label[y, x]}',
                        horizontalalignment='center',
                        verticalalignment='center',
                        )
    plt.colorbar(heatmap)

def plot_data():
    # import pickle data
    with open("PG_results.pkl", "rb") as f:
        PG_dict = pickle.load(f)

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
    plot1 = plt.figure(1)
    plt.plot(eps, np_data)

    # Labels
    plt.xlabel("Episode #")
    plt.ylabel("Simulation Reward")
    plt.title(f"Policy Gradient Rewards")

    # Save the figure as PNG
    plt.savefig(f'PG_results.png', bbox_inches='tight')
    
        
    print(PG_dict['traj'][n_episodes - 1])

    # Show the value table
    plot2 = plt.figure(2, figsize=(16, 10), dpi=80)

    # Shape the value table into a 4x12 table
    x = np.array(range(12)) + 0.5
    y = np.array(range(4)) + 0.5
    val_table = np.sum(PG_dict["value_table"], axis=1)
    val_table = np.reshape(val_table, (4,12))

    [X,Y] = np.meshgrid(x,y)

    # Build the heatmap
    make_heatmap(X,Y,val_table)
    
    # Save the figure as PNG
    plt.savefig(f'PG_table.png', bbox_inches='tight')

    action_mapping = ["RIGHT", "DOWN", "LEFT", "UP"]

    # Show Action values
    for i in range(4):
        plot_i = plt.figure(i+3, figsize=(16, 10), dpi=80)

        # Make a heatmap for actions
        action_table = PG_dict["value_table"][:,i]
        action_table = np.reshape(action_table, (4,12))
        
        make_heatmap(X,Y,action_table)
        plt.title(f"Action: {action_mapping[i]}")

        plt.savefig(f'PG_table_{action_mapping[i]}.png', bbox_inches='tight')

    # Find "best" policy
    best_actions = []
    best_action_names = []
    for i in range(48):
        a = np.argmax(PG_dict["value_table"][i,:])
        name = action_mapping[a]
        best_actions.append(PG_dict["value_table"][i,a])
        best_action_names.append(name)

    policy_table = np.array(best_actions)
    policy_table = np.reshape(policy_table, (4,12))
    policy_labels = np.array(best_action_names)
    policy_labels = np.reshape(policy_labels, (4,12))

    plt.figure(7, figsize=(16, 10), dpi=80)
    make_heatmap(X,Y,policy_table, label=policy_labels)

    plt.savefig(f'PG_table_policy.png', bbox_inches='tight')

    # Visitation heatmap
    plt.figure(8, figsize=(16, 10), dpi=80)

    visit_table = np.reshape(PG_dict["heatmap"], (4,12))
    make_heatmap(X,Y,visit_table, label=policy_labels)

    plt.savefig(f'PG_table_visit.png', bbox_inches='tight')


    # How many times did we get to the goal?
    print(f"Goal reached: {PG_dict['num_success']} times")

    # show the results
    # plt.show()
    # print(PG_dict['right_theta'])



if __name__ == "__main__":
    plot_data()