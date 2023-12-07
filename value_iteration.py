import time
import numpy as np
import torch
from SingleAgent import SingleAgent
from maze_environment import MazeEnvironment
from generate_maze import generate_maze_single_agent


def value_iteration(env, n_episodes):
    # Value Iteration algorithm to estimate state values and find an optimal policy
    gamma = 0.9  # Discount factor
    state_values = np.zeros_like(env.maze, dtype=float)

    for episode in range(n_episodes):
        state_values_new = state_values.copy()

        # Perform one-step lookahead for each state
        for row in range(env.maze.shape[0]):
            for col in range(env.maze.shape[1]):
                if env.maze[row, col] not in [0, 10]:  # Exclude walls and goal
                    values = []
                    for action in range(4):  # 4 possible actions: up, down, left, right
                        next_row, next_col = row, col
                        if action == 0:  # up
                            next_row -= 1
                        elif action == 1:  # down
                            next_row += 1
                        elif action == 2:  # left
                            next_col -= 1
                        elif action == 3:  # right
                            next_col += 1

                        # Ensure the next state is within bounds and not a wall
                        if 0 <= next_row < env.maze.shape[0] and 0 <= next_col < env.maze.shape[1] and env.maze[next_row, next_col] not in [0]:
                            values.append(state_values[next_row, next_col])

                    if values:
                        state_values_new[row, col] = env.maze[row,
                                                              col] + gamma * max(values)

        # Check for convergence (change in state values is very small)
        if np.max(np.abs(state_values_new - state_values)) < 1e-4:
            break

        state_values = state_values_new

    # Now, state_values contain the estimated values for each state
    # You can use these values to determine the policy and execute actions in the environment

    # Print the rewards earned in each episode
    for episode in range(n_episodes):
        total_rewards = 0
        state = env.reset()
        for step in range(100):  # Assuming a maximum of 100 steps per episode
            # Choose the action based on the estimated state values
            action = np.argmax([state_values[row, col]
                               for row, col in zip(*np.where(state[0] == 10))])
            next_state, reward = env.step([action])
            total_rewards += reward[0]
            state = next_state

        print(f"Episode {episode + 1}, Total Reward: {total_rewards}")
