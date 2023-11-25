import numpy as np
import random
import pygame
from generate_maze import *
from maze_environment import *

# Define SARSA parameters
ALPHA = 0.1  # learning rate
GAMMA = 0.9  # discount factor
EPSILON = 0.1  # exploration-exploitation trade-off

# Define possible actions (up, down, left, right)
ACTIONS = [0, 1, 2, 3]
NUM_ACTIONS = len(ACTIONS)

# Initialize Q-values
Q = np.zeros((16, 16, NUM_ACTIONS))

# Function to choose an action using epsilon-greedy policy
def choose_action(state):
    if np.random.rand() < EPSILON:
        return np.random.choice(NUM_ACTIONS)
    else:
        return np.argmax(Q[state])

# Function to perform SARSA update
def sarsa_update(state, action, reward, next_state, next_action):
    current_value = Q[state[0]][state[1]][action]
    next_value = Q[next_state[0]][next_state[1]][next_action]
    new_value = current_value + ALPHA * (reward + GAMMA * next_value - current_value)
    Q[state[0]][state[1]][action] = new_value

# Function to simulate one episode in the grid world
def run_episode(env):
    total_reward = 0
    state = env.reset()

    while True:
        actions = [choose_action(state[i]) for i in range(env.n_agents)]
        next_states, rewards = env.step(actions)

        for i in range(env.n_agents):
            next_state = next_states[i]
            reward = rewards[i]
            next_action = choose_action(next_state)

            sarsa_update(state[i], actions[i], reward, next_state, next_action)

            state[i] = next_state
            total_reward += reward

            if all(state[i] == env.goal_position for i in range(env.n_agents)):
                return total_reward

# Create the maze environment
maze, start_row, start_col, end_row, end_col = generate_maze()
start_positions = [(start_row, start_col)] * 3  # Assume 3 agents for simplicity
goal_position = (end_row, end_col)
env = MazeEnvironment(maze, start_positions, goal_position)

# Train the agent
NUM_EPISODES = 1000

for episode in range(NUM_EPISODES):
    total_reward = run_episode(env)

    if episode % 100 == 0:
        print(f"Episode {episode}, Total Reward: {total_reward}")

# Print the learned Q-values
print("Learned Q-values:")
print(Q)
