import numpy as np

'''
Considering this 2D matrix of 16x16 is our base

In the below maze the start and end states are represented by the value 10.

start = [7, 0]
end = [12, 15]

The numbers 1 represent the path 1.
The number 2 represent the path 2.
The number 3 represent the path 3.

To allow agents to switch paths, I have added overlaps where paths can be switched.
This switch will happen based on the transition probability at that intersection.

Aim :
3 agenst will explore these paths to find the best route. They will also communicate amongst each other so that no agent is in the same cell at the same time. This is done in-order to simulate congestion and how we can avoid it.

'''

def setup_environment():
    start = [7,0]
    end = [12, 15]


    maze = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0],
        [3, 0, 0, 3, 0, 0, 0, 0, 3, 0, 0, 3, 0, 0, 0, 0],
        [3, 0, 0, 3, 0, 0, 0, 0, 3, 0, 0, 3, 0, 0, 0, 0],
        [3, 0, 0, 3, 0, 0, 0, 0, 3, 0, 0, 3, 0, 0, 0, 0],
        [10, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 3, 0, 0, 0, 0],
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

    # print("Given Maze:")
    # print(maze)

    path_1_coords = []
    path_2_coords = []
    path_3_coords = []

    path_1_coords.append(start)
    path_2_coords.append(start)
    path_3_coords.append(start)

    # Loop through the maze array to find path coordinates
    for i in range(len(maze)):
        for j in range(len(maze[i])):
            if maze[i][j] == 1:
                path_1_coords.append((i, j))
            elif maze[i][j] == 2:
                path_2_coords.append((i, j))
            elif maze[i][j] == 3:
                path_3_coords.append((i, j))

    path_1_coords.append(end)
    path_2_coords.append(end)
    path_3_coords.append(end)

    # # Display the path coordinates
    # print("Path 1 Coordinates:")
    # print(path_1_coords)
    # print("\nPath 2 Coordinates:")
    # print(path_2_coords)
    # print("\nPath 3 Coordinates:")
    # print(path_3_coords)
    return maze, path_1_coords, path_2_coords, path_3_coords



import numpy as np
import gym
from gym import spaces

class CustomMazeEnv(gym.Env):
    def __init__(self, maze, num_agents=3):
        super(CustomMazeEnv, self).__init__()

        self.maze = maze
        self.num_rows, self.num_cols = maze.shape
        self.start_state = [(np.where(maze == 10)[0][0], np.where(maze == 10)[1][0])] * num_agents  # Start state
        self.end_state = (np.where(maze == 10)[0][1], np.where(maze == 10)[1][1])    # End state
        self.num_agents = num_agents

        self.action_space = spaces.Discrete(4)  # Up, Down, Left, Right
        self.observation_space = spaces.Tuple([spaces.Tuple((
            spaces.Discrete(self.num_rows),
            spaces.Discrete(self.num_cols)
        ))] * self.num_agents)

        self.agent_states = self.start_state

    def reset(self):
        self.agent_states = self.start_state
        return tuple(self.agent_states)

    def step(self, actions):
        if isinstance(actions, int):
            actions = [actions] * self.num_agents  # Convert a single integer to a list

        rewards = []
        new_states = []
        collisions = set()

        for i, action in enumerate(actions):
            row, col = self.agent_states[i]

            if action == 0:  # Up
                row = max(row - 1, 0)
            elif action == 1:  # Down
                row = min(row + 1, self.num_rows - 1)
            elif action == 2:  # Left
                col = max(col - 1, 0)
            elif action == 3:  # Right
                col = min(col + 1, self.num_cols - 1)

            new_state = (row, col)

            if new_state in new_states:
                collisions.add(i)

            new_states.append(new_state)

            # Check if the new state is valid
            if self.maze[row, col] == 0:  # Non-accessible state
                reward = -100
            elif self.maze[row, col] == 3:  # Path
                reward = -1
            elif self.maze[row, col] == 10:  # End state
                reward = 100
            else:
                reward = 0

            rewards.append(reward)

        self.agent_states = new_states

        # Check if any collisions occurred
        done = any(i in collisions for i in range(self.num_agents))

        return tuple(new_states), rewards, done, {}

    def render(self):
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                agent_symbols = ['A' if (i, j) == state else ' ' for state in self.agent_states]
                if any(agent_symbols):
                    print(agent_symbols, end=' ')
                elif self.maze[i, j] == 0:
                    print("X", end=' ')
                elif self.maze[i, j] == 1:
                    print("1", end=' ')
                elif self.maze[i, j] == 2:
                    print("2", end=' ')
                elif self.maze[i, j] == 3:
                    print("3", end=' ')
                elif self.maze[i, j] == 10:
                    print("S", end=' ')
            print()
        print('\n')


maze, path_1_coords, path_2_coords, path_3_coords = setup_environment()

# Create an instance of the custom environment with 3 agents
env = CustomMazeEnv(maze, num_agents=3)

# Example of using the environment
obs = env.reset()
for _ in range(20):
    actions = [env.action_space.sample() for _ in range(env.num_agents)]
    obs, rewards, done, _ = env.step(actions)
    env.render()
    if done:
        break

env.close()
