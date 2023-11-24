# Import the numpy library
import numpy as np

# Define the agent class


class SingleAgent:
    # Initialize the agent with its position, maze, and reward
    def __init__(self, position, maze, reward):
        self.position = position  # A tuple of (row, col)
        self.maze = maze  # A 2D array of integers
        self.reward = reward  # A dictionary of cell values and rewards

    # Define a method to check if the agent has reached the goal
    def is_goal(self):
        return self.maze[self.position[0]][self.position[1]] == 10

    # Define a method to get the possible actions for the agent
    def get_actions(self):
        # The agent can move up, down, left, or right
        actions = ["up", "down", "left", "right"]
        # Get the current row and column of the agent
        row, col = self.position
        # Remove the actions that would lead to invalid or blocked cells
        if row == 0 or self.maze[row - 1][col] == 0:
            actions.remove("up")
        if row == 15 or self.maze[row + 1][col] == 0:
            actions.remove("down")
        if col == 0 or self.maze[row][col - 1] == 0:
            actions.remove("left")
        if col == 15 or self.maze[row][col + 1] == 0:
            actions.remove("right")
        # Return the remaining actions
        return actions

    # Define a method to perform an action and update the agent's position
    def do_action(self, action):
        # Get the current row and column of the agent
        row, col = self.position
        # Update the row and column based on the action
        if action == "up":
            row -= 1
        elif action == "down":
            row += 1
        elif action == "left":
            col -= 1
        elif action == "right":
            col += 1
        # Set the new position of the agent
        self.position = (row, col)

    # Define a method to get the reward for the agent's current position
    def get_reward(self):
        # Get the value of the cell at the agent's position
        cell_value = self.maze[self.position[0]][self.position[1]]
        # Return the reward for that value from the reward dictionary
        return self.reward[cell_value]
