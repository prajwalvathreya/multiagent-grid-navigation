import numpy as np
import pygame
import matplotlib.pyplot as plt
from tqdm import tqdm

class MazeEnvironment:
    def __init__(self, maze, current_state, goal_states):
        """
        MazeEnvironment constructor.

        Parameters:
        - maze (numpy.ndarray): The maze grid.
        - current_state (list): Initial state of the agent [row, column].
        - goal_states (list): List of goal states [row, column].
        """
        self.maze = maze
        self.current_state = current_state
        self.goal_states = goal_states
        self.collided = False
        self.agent_path = []  # Store the path taken by the agent
        pygame.init()

    def move(self, action):
        """
        Move the agent based on the given action.

        Parameters:
        - action (int): The action to take (0: up, 1: down, 2: left, 3: right).
        """
        row, col = self.current_state
        if action == 0 and row > 0 and self.maze[row - 1, col]:
            self.current_state = (row - 1, col)
        elif action == 1 and row < self.maze.shape[0] - 1 and self.maze[row + 1, col]:
            self.current_state = (row + 1, col)
        elif action == 2 and col > 0 and self.maze[row, col - 1]:
            self.current_state = (row, col - 1)
        elif action == 3 and col < self.maze.shape[1] - 1 and self.maze[row, col + 1]:
            self.current_state = (row, col + 1)

        self.agent_path.append(np.ravel_multi_index(self.current_state, self.maze.shape))

    def get_reward(self):
        """
        Get the reward based on the current state.

        Returns:
        - float: The reward value.
        """
        row, col = self.current_state
        if self.current_state == self.goal_states:
            return 10  # Goal reached
        elif self.maze[row, col] in [1, 2, 3]:
            return 0.1  # Normal path
        elif self.maze[row, col] == 5:
            return 0  # Start state
        elif self.collided:
            return -100  # Obstacle or invalid state
        else:
            if np.ravel_multi_index(self.current_state, self.maze.shape) not in self.agent_path:
                return 0.1  # Small positive reward for exploration
            else:
                return 0  # Already visited this state, no additional reward

    def is_goal_state(self):
        """
        Check if the current state is a goal state.

        Returns:
        - bool: True if the current state is a goal state, False otherwise.
        """
        return self.current_state in self.goal_states

    def has_collided(self):
        """
        Check if the agent has collided with an obstacle.

        Returns:
        - bool: True if the agent has collided, False otherwise.
        """
        return self.collided

    def render(self, size=600):
        """
        Render the maze environment using Pygame.

        Parameters:
        - size (int): Size of the Pygame window.
        """
        width, height = self.maze.shape
        cell_size = size // max(width, height)
        screen = pygame.display.set_mode((width * cell_size, height * cell_size))
        pygame.display.set_caption("Maze Navigation")

        colors = {
            0: (0, 0, 0),
            1: (255, 0, 0),
            2: (0, 255, 0),
            3: (0, 0, 255),
            5: (255, 255, 0),
            9: (0, 255, 0),
            10: (255, 0, 0)
        }

        screen.fill((0, 0, 0))

        for row in range(height):
            for col in range(width):
                rect = pygame.Rect(col * cell_size, row * cell_size, cell_size, cell_size)
                cell_value = self.maze[row, col]
                pygame.draw.rect(screen, colors[cell_value], rect)

        path_color = (0, 255, 255)

        for state in self.agent_path:
            row, col = np.unravel_index(state, self.maze.shape)
            pygame.draw.rect(screen, path_color, (col * cell_size, row * cell_size, cell_size, cell_size))

        row, col = self.current_state
        center_x = col * cell_size + cell_size // 2
        center_y = row * cell_size + cell_size // 2
        radius = cell_size // 4

        # Define vertices of the triangle
        vertices = [
            (center_x, center_y - radius),  # Top vertex
            (center_x - radius * np.sin(np.radians(60)), center_y + radius * np.cos(np.radians(60))),  # Bottom left vertex
            (center_x + radius * np.sin(np.radians(60)), center_y + radius * np.cos(np.radians(60)))   # Bottom right vertex
        ]

        pygame.draw.polygon(screen, (255, 255, 255), vertices)

        pygame.display.flip()

        pygame.quit()

def q_learning(env, q_table, num_episodes=1000, initial_alpha=0.9, gamma=0.9, epsilon=0.9, epsilon_decay=0.9999):
    
    """
    Q-learning algorithm for training an agent to navigate the maze.

    Parameters:
    - env (MazeEnvironment): The environment in which the agent operates.
    - q_table (numpy.ndarray): The Q-table representing the expected cumulative future rewards for each state-action pair.
    - num_episodes (int): Number of training episodes.
    - initial_alpha (float): Initial learning rate.
    - gamma (float): Discount factor for future rewards.
    - epsilon (float): Exploration-exploitation trade-off parameter.
    - epsilon_decay (float): Rate at which epsilon decays over episodes.

    Returns:
    - q_table (numpy.ndarray): Updated Q-table after training.
    - rewards_per_episode (list): List of total rewards obtained in each episode.
    """
    rewards_per_episode = []

    for episode in tqdm(range(num_episodes)):
        # reset environment
        env.current_state = [6, 0]
        state = np.ravel_multi_index(env.current_state, env.maze.shape)
        total_reward = 0

        alpha = initial_alpha / (1 + episode)

        while not env.is_goal_state():
            if np.random.rand() < epsilon:
                action = np.random.choice(4)
            else:
                max_actions = np.argwhere(q_table[state] == np.max(q_table[state])).flatten()
                action = np.min(max_actions)

            env.move(action)

            next_state = np.ravel_multi_index(env.current_state, env.maze.shape)
            reward = env.get_reward()

            next_action = np.argmax(q_table[next_state])

            q_table[state, action] = (1 - alpha) * q_table[state, action] + alpha * (
                    reward + gamma * q_table[next_state, next_action])

            state = next_state
            total_reward += reward

            if env.has_collided:
                break

        epsilon *= epsilon_decay

        rewards_per_episode.append(total_reward)

        # maze_env.render()

    return q_table, rewards_per_episode

# Example maze for testing
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

maze_env = MazeEnvironment(maze=maze, current_state=[6, 0], goal_states=[12, 15])

num_actions = 4
num_states = 256
q_table = np.zeros((num_states, num_actions))

q_table, rewards = q_learning(maze_env, q_table)

# Plotting the rewards over episodes
plt.plot(rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Reward over Episodes')
plt.show()
