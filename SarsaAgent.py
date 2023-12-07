# SARSA Agent class
import numpy as np
from SingleAgent import SingleAgent  # Assuming SingleAgent is a defined class


class SarsaAgent(SingleAgent):
    def __init__(self, position, maze, reward):
        super().__init__(position, maze, reward)
        # Initialize Q-values for each state-action pair
        self.q_values = np.zeros(
            (maze.shape[0], maze.shape[1], len(self.get_actions())))

    def get_q_value(self, state, action):
        row, col = state[0], state[1]  # Extract row and column from the state
        action_index = self.get_actions().index(action)
        return self.q_values[row, col, action_index]

    def update_q_value(self, state, action, value):
        row, col = state[0], state[1]  # Extract row and column from the state
        action_index = self.get_actions().index(action)
        # Update the Q-value for the specified state-action pair
        self.q_values[row, col, action_index] += value

    def get_best_action(self, state):
        row, col = state[0], state[1]  # Extract row and column from the state
        action_index = np.argmax(self.q_values[row, col])
        # Return the action corresponding to the highest Q-value
        return self.get_actions()[action_index]
