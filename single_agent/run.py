import numpy as np
from dqn_agent import DQNAgent
from single_agent_environment import MazeEnvironment
import torch
from tqdm import tqdm

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

device = "mps" if torch.backends.mps.is_available else "cpu"

start_position = (6, 0)
goal_position = (12, 15)

env = MazeEnvironment(maze, start_position, goal_position)

# Define DQN parameters
state_size = 16 * 16  # Modify according to your state representation
action_size = 4  # Modify according to the number of discrete actions
agent = DQNAgent(state_size, action_size, device)

# Training loop parameters
num_episodes = 1000
batch_size = 64

max_steps = 100

for episode in tqdm(range(num_episodes)):
    state = env.reset()
    total_reward = 0
    done = False  # Flag to track episode termination
    visited_positions = set()  # Set to store visited positions

    for step in range(max_steps):
        action = agent.choose_action(state)
        next_state, reward = env.step(action)

        # Check if the next position is already visited
        if tuple(env.position) in visited_positions:
            reward = -10  # Penalize revisiting a cell

        visited_positions.add(tuple(env.position))  # Add the current position to the set

        agent.remember(state, action, reward, next_state, done)
        agent.replay(batch_size)

        state = next_state
        total_reward += reward

        # env.render()

        if done:
            break  # Break if the environment signals episode termination

    agent.update_target_network()

    # Print total reward for the episode
    print(f"Episode {episode + 1}, Total Reward: {total_reward}")

    # Save the trained model if needed
    if (episode + 1) % 100 == 0:
        torch.save(agent.q_network.state_dict(), f'model_state_dicts/trained_model_{episode + 1}.pth')

# Save the trained model if needed
torch.save(agent.q_network.state_dict(), 'trained_model.pth')
