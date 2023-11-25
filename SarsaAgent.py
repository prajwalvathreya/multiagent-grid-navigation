# SARSA Agent class
import numpy as np
from SingleAgent import SingleAgent


class SarsaAgent(SingleAgent):
    def __init__(self, position, maze, reward):
        super().__init__(position, maze, reward)
        self.q_values = np.zeros(
            (maze.shape[0], maze.shape[1], len(self.get_actions())))

    def get_q_value(self, state, action):
        row, col = state[0], state[1]  # Remove the .item() method calls
        action_index = self.get_actions().index(action)
        return self.q_values[row, col, action_index]

    def update_q_value(self, state, action, value):
        row, col = state[0], state[1]  # Remove the .item() method calls
        action_index = self.get_actions().index(action)
        self.q_values[row, col, action_index] += value

    def get_best_action(self, state):
        print(state)
        print('----')
        row, col = state[0], state[1]  # Remove the .item() method calls
        action_index = np.argmax(self.q_values[row, col])
        return self.get_actions()[action_index]
