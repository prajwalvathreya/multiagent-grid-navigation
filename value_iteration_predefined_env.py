import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def value_iteration(env, gamma=0.9, epsilon=1e-6):
    # Value Iteration algorithm to find optimal values for states and rewards
    num_states = env.observation_space.n
    num_actions = env.action_space.n

    # Initialize value function
    V = np.zeros(num_states)

    # Initialize list to store rewards at each iteration
    rewards_list = []

    while True:
        delta = 0
        total_reward = 0

        for s in range(num_states):
            v = V[s]
            # Compute the Bellman update using the maximum expected future reward
            V[s] = max([sum([p * (r + gamma * V[s_]) for p, s_, r,
                       _ in env.P[s][a]]) for a in range(num_actions)])
            delta = max(delta, abs(v - V[s]))

        # Calculate the total reward for the current policy (assuming the reward is the value of the state)
        for s in range(num_states):
            total_reward += V[s]

        rewards_list.append(total_reward)

        if delta < epsilon:
            break

    return V, rewards_list




def plot_policy(env, policy):
    # Plot the optimal policy on the environment grid
    grid = np.zeros(env.desc.shape, dtype=int)
    for s in range(len(policy)):
        row, col = env.decode(s)
        grid[row, col] = policy[s]

    cmap = ListedColormap(['white', 'black', 'blue', 'green'])
    plt.imshow(grid, cmap=cmap, interpolation='none')
    plt.show()



if __name__ == "__main__":
    # Assuming you have the OpenAI Gym library installed
    import gym

    # Create FrozenLake environment
    env = gym.make('FrozenLake8x8-v1')

    # Run value iteration
    optimal_values, rewards_list = value_iteration(env)
    print(rewards_list)
    # Plot the optimal values
    plt.plot(optimal_values)
    plt.xlabel('State')
    plt.ylabel('Optimal Value')
    plt.title('Value Iteration Performance')
    plt.show()

    # Plot the rewards over iterations
    plt.plot(rewards_list)
    plt.xlabel('Iteration')
    plt.ylabel('Total Reward')
    plt.title('Rewards during Value Iteration')
    plt.show()

    # Find and plot the optimal policy
    optimal_policy = [np.argmax([sum([p * (r + 0.9 * optimal_values[s_]) for p, s_, r, _ in env.P[s][a]]) for a in range(env.action_space.n)]) for s in range(env.observation_space.n)]
    print(optimal_policy)
    optimal_policy = [np.argmax([sum([p * (r + 0.9 * optimal_values[s_]) for p, s_, r, _ in env.P[s][a]])
                                for a in range(env.action_space.n)]) for s in range(env.observation_space.n)]
    plot_policy(env, optimal_policy)
