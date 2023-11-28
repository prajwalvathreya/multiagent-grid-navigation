import numpy as np
from gym import Env, spaces

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3


class CustomFrozenLakeEnv(Env):
    def __init__(self, map_size=4, is_slippery=True):
        self.map_size = map_size
        self.is_slippery = is_slippery

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(map_size * map_size)

        self.board = self.generate_random_map()

        self.current_row = 0
        self.current_col = 0

    def generate_random_map(self, p=0.8):
        valid = False
        while not valid:
            p = min(1, p)
            board = np.random.choice(
                ["F", "H"], (self.map_size, self.map_size), p=[p, 1 - p])
            board[0][0] = "S"
            board[-1][-1] = "G"
            valid = self.is_valid(board)
        return board

    def is_valid(self, board):
        # Implement DFS to check if there is a path from start to goal
        frontier, discovered = [], set()
        frontier.append((0, 0))
        while frontier:
            r, c = frontier.pop()
            if not (r, c) in discovered:
                discovered.add((r, c))
                directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
                for x, y in directions:
                    r_new = r + x
                    c_new = c + y
                    if r_new < 0 or r_new >= self.map_size or c_new < 0 or c_new >= self.map_size:
                        continue
                    if board[r_new][c_new] == "G":
                        return True
                    if board[r_new][c_new] != "H":
                        frontier.append((r_new, c_new))
        return False

    def step(self, action):
        if self.is_slippery:
            # Implement slippery logic
            if np.random.rand() < 0.2:
                action = np.random.choice([LEFT, DOWN, RIGHT, UP])
        else:
            # Implement non-slippery logic
            pass

        # Update the state and get the reward
        if action == LEFT:
            self.current_col = max(self.current_col - 1, 0)
        elif action == DOWN:
            self.current_row = min(self.current_row + 1, self.map_size - 1)
        elif action == RIGHT:
            self.current_col = min(self.current_col + 1, self.map_size - 1)
        elif action == UP:
            self.current_row = max(self.current_row - 1, 0)

        state = self.current_row * self.map_size + self.current_col
        reward = self.get_reward()
        done = self.is_done()

        return state, reward, done, {}

    def get_reward(self):
        if self.board[self.current_row][self.current_col] == "G":
            return 1
        elif self.board[self.current_row][self.current_col] == "H":
            return 0
        else:
            return 0

    def is_done(self):
        return self.board[self.current_row][self.current_col] in ["G", "H"]

    def reset(self):
        # Reset the environment to the initial state
        self.current_row = 0
        self.current_col = 0
        return 0

    def render(self, mode='human'):
        # Implement rendering logic if needed
        # For simplicity, let's just print the current state
        print(self.board)
        print("Current Position: ({}, {})".format(
            self.current_row, self.current_col))

    def close(self):
        # Implement any cleanup logic
        pass


# Example usage
env = CustomFrozenLakeEnv(map_size=4, is_slippery=True)
state = env.reset()

for _ in range(10):
    action = env.action_space.sample()
    next_state, reward, done, _ = env.step(action)

    print(
        f"Action: {action}, Next State: {next_state}, Reward: {reward}, Done: {done}")

    env.render()

    if done:
        print("Episode finished!")
        break

env.close()
