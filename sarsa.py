import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque
import numpy as np

# Define the grid world
GRID_SIZE = 5
START_STATE = (0, 0)
GOAL_STATE = (GRID_SIZE-1, GRID_SIZE-1)
OBSTACLES = [(1, 1), (2, 2), (3, 3)]  # You can customize the obstacle positions

# Define possible actions (up, down, left, right)
ACTIONS = [(0, 1), (0, -1), (1, 0), (-1, 0)]
NUM_ACTIONS = len(ACTIONS)

# Define SARSA parameters
ALPHA = 0.1  # learning rate
GAMMA = 0.9  # discount factor
EPSILON = 0.1  # exploration-exploitation trade-off

# Initialize Q-values
Q = np.zeros((GRID_SIZE, GRID_SIZE, NUM_ACTIONS))

# Function to choose an action using epsilon-greedy policy
def choose_action(state):
    if np.random.rand() < EPSILON:
        return np.random.choice(NUM_ACTIONS)
    else:
        return np.argmax(Q[state])

# Function to perform SARSA update
def sarsa_update(state, action, reward, next_state, next_action):
    current_value = Q[state][action]
    next_value = Q[next_state][next_action]
    new_value = current_value + ALPHA * (reward + GAMMA * next_value - current_value)
    Q[state][action] = new_value

# Function to simulate one episode in the grid world
def run_episode():
    total_reward = 0
    current_state = START_STATE
    current_action = choose_action(current_state)

    while current_state != GOAL_STATE:
        next_state = (current_state[0] + ACTIONS[current_action][0], current_state[1] + ACTIONS[current_action][1])

        # Check if the next state is valid (not outside the grid or an obstacle)
        if 0 <= next_state[0] < GRID_SIZE and 0 <= next_state[1] < GRID_SIZE and next_state not in OBSTACLES:
            # Calculate reward (negative for each step until reaching the goal)
            reward = -1
            if next_state == GOAL_STATE:
                reward = 0

            next_action = choose_action(next_state)
            sarsa_update(current_state, current_action, reward, next_state, next_action)

            current_state = next_state
            current_action = next_action
            total_reward += reward

    return total_reward

# Train the agent
NUM_EPISODES = 1000

for episode in range(NUM_EPISODES):
    total_reward = run_episode()

    if episode % 100 == 0:
        print(f"Episode {episode}, Total Reward: {total_reward}")

# Print the learned Q-values
print("Learned Q-values:")
print(Q)
