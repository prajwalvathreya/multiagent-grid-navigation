import numpy as np
from single_agent_environment import MazeEnvironment  # Import your MazeEnvironment class
from dqnagent import DQNAgent  # Import your DQNAgent class
import torch

def inference(model_path, maze, start_position, goal_position):
    # Create the maze environment
    env = MazeEnvironment(maze, start_position, goal_position)

    # Define DQN parameters
    state_size = 16 * 16  # Modify according to your state representation
    action_size = 4  # Modify according to the number of discrete actions
    device = "cpu" if torch.backends.mps.is_available else "cpu"

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
            state_tensor = torch.tensor(state, dtype=torch.float32).view(1, -1).to(device)
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
    model_path = 'trained_model.pth'  # Update with the correct path

    # Define your maze, start position, and goal position
    maze = np.array([
        # ... (your maze definition)
    ])

    start_position = (6, 0)  # Update with the correct start position
    goal_position = (12, 15)  # Update with the correct goal position

    # Perform inference
    inference(model_path, maze, start_position, goal_position)
