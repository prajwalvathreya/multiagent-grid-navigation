import numpy as np
import pygame

class MazeEnvironment:
    def __init__(self, maze, start_positions, goal_position):
        self.maze = maze
        self.start_positions = start_positions
        self.goal_position = goal_position
        self.n_agents = len(start_positions)
        self.positions = np.array(start_positions)
        self.done = False

    def step(self, actions):

        rewards = np.zeros(self.n_agents)
        next_states = np.zeros((self.n_agents, *self.maze.shape))
        self.done = False

        for i in range(self.n_agents):
            current_position = self.positions[i]
            self.update_position(i, actions[i])
            new_position = self.positions[i]

            if self.is_out_of_bounds(new_position):
                rewards[i] -= 100  # Penalize for going out of bounds
                self.positions[i] = current_position
                self.done = True

            elif self.is_collision(i) and not self.is_at_allowed_shared_cell(i):
                rewards[i] -= 100
                self.done = True
                
            elif self.maze[tuple(new_position)] in [1, 2, 3]:
                # Reward for visiting a new cell
                rewards[i] += 1 if any(new_position != current_position) else 0
                self.done = False

            elif self.maze[tuple(new_position)] in [10]:
                # Reward for visiting a new cell
                rewards[i] += 1000 if any(new_position != current_position) else 0
                self.done = True

            else:
                rewards[i] -= 100
                self.done = True

            next_states[i] = self.get_agent_state(i)

        return next_states, rewards, self.done

    def update_position(self, agent_index, action):
        # Define actions as up (0), down (1), left (2), right (3)
        action_map = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        new_position = self.positions[agent_index] + action_map[action]

        if 0 <= new_position[0] < self.maze.shape[0] and \
           0 <= new_position[1] < self.maze.shape[1] and \
           self.maze[tuple(new_position)] in [1, 2, 3, 10]:  # Include 10 for valid moves
            self.positions[agent_index] = new_position

    def is_collision(self, agent_index):
        for i in range(self.n_agents):
            if i != agent_index and all(self.positions[i] == self.positions[agent_index]):
                return True
        return False

    def is_at_allowed_shared_cell(self, agent_index):
        return all(self.positions[agent_index] == self.start_positions[agent_index]) or \
            all(self.positions[agent_index] == self.goal_position)

    def get_agent_state(self, agent_index):
        state = np.copy(self.maze)
        for i, position in enumerate(self.positions):
            if i != agent_index:
                state[tuple(position)] = -1  # Mark other agents' positions
        return state

    def reset(self):
        self.positions = np.array(self.start_positions)
        return [self.get_agent_state(i) for i in range(self.n_agents)]
    
    def is_out_of_bounds(self, position):
        return (
            position[0] < 0 or position[0] >= self.maze.shape[0] or
            position[1] < 0 or position[1] >= self.maze.shape[1]
        )

    def render(self, size=600):

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
                elif cell_value == 10:
                    pygame.draw.rect(screen, (255, 255, 255), rect)

        for position in self.positions:
            center_x = position[1] * cell_size + cell_size // 2
            center_y = position[0] * cell_size + cell_size // 2
            radius = cell_size // 4  # Radius of the circle in which the triangle is inscribed

            # Define vertices of the triangle
            vertices = [
                (center_x, center_y - radius),  # Top vertex
                (center_x - radius * np.sin(np.radians(60)), center_y + \
                 radius * np.cos(np.radians(60))),  # Bottom left vertex
                (center_x + radius * np.sin(np.radians(60)), center_y + \
                 radius * np.cos(np.radians(60)))   # Bottom right vertex
            ]

            pygame.draw.polygon(screen, (255, 255, 255), vertices)

        pygame.display.flip()
        pygame.time.wait(100)
