import time
import numpy as np
import torch
from SingleAgent import SingleAgent
from maze_environment import MazeEnvironment
from generate_maze import generate_maze_single_agent
from dfs import *

# Define the maze
maze, start_row, start_col, end_row, end_col = generate_maze_single_agent()

# Initialize the environment and the agent
# Create a single agent for the maze with the given reward
reward = {0: -100, 1: 0, 2: 0, 3: 0, 10: 100}
agent = SingleAgent((start_row, start_col), maze, reward)
env = MazeEnvironment(maze=maze, start_positions=[
                      [start_row, start_col]], goal_position=[end_row, end_col])
print(maze)
print(env.render())
time.sleep(10) # Just for testing

# Set parameters for training
n_episodes = 1000
dfs(agent,env,n_episodes)
