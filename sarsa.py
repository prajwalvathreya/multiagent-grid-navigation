import time
import numpy as np
import torch
from SarsaAgent import SarsaAgent
from SarsaAlgo import sarsa
from maze_environment import MazeEnvironment
from generate_maze import generate_maze_single_agent
from dfs import *
from value_iteration import *
import matplotlib.pyplot as plt

# Set the learning rate, discount factor, and number of episodes (hardcoded values)
lr = 0.8
gamma = 0.95
num_episodes = 1000

# Initialize the environment
maze, start_row, start_col, end_row, end_col = generate_maze_single_agent()
env = MazeEnvironment(maze, [(start_row, start_col)], (end_row, end_col))

# Initialize the Q-table
n_actions = 4  # Assuming four possible actions: up, down, left, right
n_states = env.maze.size  # Assuming the state is represented as a single integer
Q = np.random.uniform(low=0, high=0.01, size=(n_states, n_actions))

# Lists to store performance metrics
total_rewards = []
total_steps = []
success_rate = []

# Run the SARSA algorithm
for i in range(num_episodes):
    s = env.reset()
    done = False
    episode_reward = 0
    num_steps = 0

    # Choose the first action using epsilon-greedy policy
    a = np.argmax(Q[s, :] + np.random.randn(1, n_actions)[0] * (1. / (i + 1)))

    # The SARSA learning algorithm
    while not done:
        s_, r, done, _, _ = env.step(a)

        # Choose the next action using epsilon-greedy policy
        a_ = np.argmax(Q[s_, :] + np.random.randn(1, n_actions)[0] * (1. / (i + 1)))

        # Update Q values using SARSA update rule
        Q[s, a] = Q[s, a] + lr * (r + gamma * Q[s_, a_] - Q[s, a])

        s = s_
        a = a_
        episode_reward += r
        num_steps += 1

    # Append performance metrics to lists
    total_rewards.append(episode_reward / num_steps)
    total_steps.append(num_steps)
    success_rate.append(int(episode_reward > 0))

    # Print episode metrics
    print("Episode:", i + 1, "Reward:", episode_reward, "Steps:", num_steps)

# Plotting the average rewards and episode lengths gained throughout each episode
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].plot(total_rewards, 'tab:green')
axs[0].set_title('Average Reward per Episode')
axs[1].plot(total_steps, 'tab:purple')
axs[1].set_title('Number of Steps Taken per Episode')

plt.show()

# Print overall metrics
print('----------------------------------------------------------')
print("Overall Average reward:", np.mean(total_rewards))
print("Overall Average number of steps:", np.mean(total_steps))
print("Success rate (%):", np.mean(success_rate) * 100)