import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque
import numpy as np
from maze_environment_multiagent import *
from dqnagent import *

# Define the maze layout
maze = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0],
    [3, 0, 0, 3, 0, 0, 0, 0, 3, 0, 0, 3, 0, 0, 0, 0],
    [3, 0, 0, 3, 0, 0, 0, 0, 3, 0, 0, 3, 0, 0, 0, 0],
    [3, 0, 0, 3, 0, 0, 0, 0, 3, 0, 0, 3, 0, 0, 0, 0],
    [5, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 3, 0, 0, 0, 0],
    [2, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 3, 0, 0, 0, 0],
    [2, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 3, 0, 0, 0, 0],
    [2, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 3, 0, 0, 0, 0],
    [2, 2, 2, 1, 2, 2, 2, 2, 1, 2, 2, 3, 3, 3, 3, 3],
    [2, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 3],
    [2, 2, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 10],
    [0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
])

# Define starting positions for each agent and the goal position
start_positions = [(6, 0), (6, 0), (6, 0)]
goal_position = (12, 15)

# Create a MazeEnvironment instance with the defined maze and positions
env = MazeEnvironment(maze, start_positions, goal_position)

# Initialize multiple DQNAgents
n_agents = env.n_agents
state_size = env.maze.shape
action_size = 4  # Assuming 4 possible actions (up, down, left, right)
device = "cpu"

# Create a list of DQNAgents, one for each agent in the environment
agents = [DQNAgent(state_size, action_size, device) for _ in range(n_agents)]

# Training parameters
num_episodes = 1000
max_steps_per_episode = 100
batch_size = 32
target_update_frequency = 10

# Training loop over episodes
for episode in range(num_episodes):
    # Reset the environment and get the initial state for each agent
    states = env.reset()
    episode_done = False  # Initialize episode_done flag

    # Step through the environment for a maximum number of steps per episode
    for step in range(max_steps_per_episode):
        # Choose actions for each agent using their DQN policies
        actions = [agent.choose_action(state)
                   for agent, state in zip(agents, states)]

        # Take actions in the environment and observe next states and rewards
        next_states, rewards, episode_done = env.step(actions)

        # Remember the experience for each agent and perform replay
        for i, agent in enumerate(agents):
            agent.remember(states[i], actions[i],
                           rewards[i], next_states[i], False)
            agent.replay(batch_size)

        states = next_states
        env.render()  # Optionally render the environment

        if episode_done:
            break  # End the episode if any agent collides

    # Optionally, update target networks periodically
    if episode % target_update_frequency == 0:
        for agent in agents:
            agent.update_target_network()
