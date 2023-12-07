import numpy as np
import pygame


class MazeEnvironment:

    def __init__(self, maze, start_position, goal_position, number):
        # Initialize the MazeEnvironment with the given maze, start_position, goal_position, and number
        self.maze = maze
        self.start_position = start_position
        self.goal_position = goal_position
        self.position = np.array(start_position)
        self.done = False
        self.number = number

    def step(self, action):
        # Take a step in the environment based on the provided action
        action_map = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        reward = 0
        new_position = self.position + action_map[action]

        # Check if the new position is valid
        if (
            0 <= new_position[0] < self.maze.shape[0]
            and 0 <= new_position[1] < self.maze.shape[1]
            and self.maze[tuple(new_position)] in [1, 2, 3, 10]
        ):
            self.position = new_position
            self.done = False
            reward = 0
        elif (
            0 <= new_position[0] < self.maze.shape[0]
            and 0 <= new_position[1] < self.maze.shape[1]
            and self.maze[tuple(new_position)] in [5]
        ):
            self.position = new_position
            self.done = False
            reward = -200  # Penalize if agent goes back to the start position
        else:
            reward = -200  # Penalize if the agent chooses a location outside the matrix or goes out of bounds or a cell with 0
            self.done = True

        # Check if the goal is reached
        if tuple(self.position) == tuple(self.goal_position):
            print("Goal Reached")
            reward = 3000  # Positive reward for reaching the goal
            self.done = True

        return self.get_agent_state(), reward

    def get_agent_state(self):
        # Get the current state of the agent in the environment
        state = np.zeros_like(self.maze)
        state[tuple(self.position)] = 1  # Mark the agent's position
        return state

    def reset(self):
        # Reset the environment to the starting state
        self.position = np.array(self.start_position)
        self.done = False
        return self.get_agent_state()

    def update_position(self, agent_index, action):
        # Update the position of the agent based on the provided action
        action_map = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}

        if isinstance(action, list):
            action = action[0]  # Unwrap action if it's a list

        new_position = self.position + action_map[action]

        # Check if the new position is valid
        if (
            0 <= new_position[0] < self.maze.shape[0]
            and 0 <= new_position[1] < self.maze.shape[1]
            # Include 10 for valid moves
            and self.maze[tuple(new_position)] in [1, 2, 3, 10]
        ):
            self.position = new_position

    def render(self, size=600):
        # Render the current state of the environment using Pygame
        pygame.init()
        width, height = self.maze.shape
        cell_size = size // max(width, height)
        screen = pygame.display.set_mode(
            (width * cell_size, height * cell_size))
        pygame.display.set_caption("Maze Navigation")

        # Define colors
        colors = {1: (255, 0, 0),  # Red for path '1'
                  2: (0, 255, 0),  # Green for path '2'
                  3: (0, 0, 255)}  # Blue for path '3'

        for row in range(height):
            for col in range(width):
                rect = pygame.Rect(col * cell_size, row *
                                   cell_size, cell_size, cell_size)
                cell_value = self.maze[row, col]
                if cell_value == 0:  # Wall
                    pygame.draw.rect(screen, (0, 0, 0), rect)
                elif cell_value in [1, 2, 3]:  # Paths
                    pygame.draw.rect(screen, colors[cell_value], rect)
                elif cell_value == 10 or cell_value == 5:
                    pygame.draw.rect(screen, (255, 255, 255), rect)

        center_x = self.position[1] * cell_size + cell_size // 2
        center_y = self.position[0] * cell_size + cell_size // 2
        radius = cell_size // 4  # Radius of the circle in which the triangle is inscribed

        # Define vertices of the triangle
        vertices = [
            (center_x, center_y - radius),  # Top vertex
            (center_x - radius * np.sin(np.radians(60)), center_y + \
             radius * np.cos(np.radians(60))),  # Bottom left vertex
            (center_x + radius * np.sin(np.radians(60)), center_y + \
             radius * np.cos(np.radians(60)))   # Bottom right vertex
        ]

        pygame.draw.polygon(screen, (0, 0, 0), vertices)

        pygame.display.flip()
        pygame.time.wait(100)

    def save_environment(self, size=600):
        # Save the rendered environment as an image
        pygame.init()
        width, height = self.maze.shape
        cell_size = size // max(width, height)
        screen = pygame.Surface((width * cell_size, height * cell_size))

        # Define colors
        colors = {1: (255, 0, 0),  # Red for path '1'
                  2: (0, 255, 0),  # Green for path '2'
                  3: (0, 0, 255)}  # Blue for path '3'

        for row in range(height):
            for col in range(width):
                rect = pygame.Rect(col * cell_size, row *
                                   cell_size, cell_size, cell_size)
                cell_value = self.maze[row, col]
                if cell_value == 0:  # Wall
                    pygame.draw.rect(screen, (0, 0, 0), rect)
                elif cell_value in [1, 2, 3]:  # Paths
                    pygame.draw.rect(screen, colors[cell_value], rect)
                elif cell_value == 10 or cell_value == 5:
                    pygame.draw.rect(screen, (255, 255, 255), rect)

        pygame.image.save(
            screen, f"environments/environment_image_{self.number}.png")

    def terminated(self):
        # Check if the environment has terminated
        return self.done
