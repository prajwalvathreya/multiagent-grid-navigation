import numpy as np
import torch
# Import your MazeEnvironment class
from single_agent_environment import MazeEnvironment
from dqnagent import DQNAgent  # Import your DQNAgent class
from mazegenerator import *


def inference(model_path, maze, start_position, goal_position):
    # Create the maze environment
    env = MazeEnvironment(maze, start_position, goal_position, 0)

    # Define DQN parameters
    state_size = 16 * 16  # Assuming state is a flattened 16x16 grid
    action_size = 4  # Assuming 4 possible actions (up, down, left, right)
    device = "cpu"

    # Initialize the DQNAgent
    agent = DQNAgent(state_size, action_size, device)

    # Load the trained model
    agent.q_network.load_state_dict(torch.load(model_path))
    agent.q_network.eval()

    # Inference loop
    state = env.reset()
    done = False

    while not done:
        # Choose an action using the trained model
        with torch.no_grad():
            state_tensor = torch.tensor(
                state, dtype=torch.float32).view(1, -1).to(device)
            q_values = agent.q_network(state_tensor)
            action = torch.argmax(q_values).item()

        # Take the chosen action
        next_state, reward = env.step(action)
        state = next_state

        # Check if the episode is done
        done = (tuple(env.position) == tuple(goal_position))

        # Render the environment (optional)
        env.render()

    print("Inference complete.")


if __name__ == "__main__":
    # Path to the trained model
    model_path = r'trained_model.pth'  # Update with the correct path

    # Define your maze, start position, and goal position
    # maze, s_row, s_col, g_row, g_col = generate_maze()

    # start_position = (s_row, s_col)
    # goal_position = (g_row, g_col)
    # Create the maze environment
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

    start_position = (6, 0)
    goal_position = (12, 15)

    # Perform inference
    inference(model_path, maze, start_position, goal_position)
