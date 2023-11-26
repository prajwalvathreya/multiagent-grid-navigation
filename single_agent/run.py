import numpy as np
from dqn_agent import DQNAgent
from single_agent_environment import MazeEnvironment
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

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

device = "cpu" if torch.backends.mps.is_available else "cpu"

start_position = (6, 0)
goal_position = (12, 15)

env = MazeEnvironment(maze, start_position, goal_position)

# Define DQN parameters
state_size = 16 * 16  # Modify according to your state representation
action_size = 4  # Modify according to the number of discrete actions
agent = DQNAgent(state_size, action_size, device)

# Training loop parameters
num_episodes = 1000
batch_size = 32
max_steps = 100
loss_every_episode = []
reward_every_episode = []
action_map = {0: "up", 1: "down", 2: "left", 3: "right"}
for episode in tqdm(range(num_episodes)):
    state = env.reset()
    total_reward = 0
    visited_positions = set()  # Set to store visited positions
    total_loss = 0
    done = False

    while not env.terminated():

        action = agent.choose_action(state)
        # print(action_map[action])
        next_state, reward = env.step(action)

        # Check if the next position is already visited
        if tuple(env.position) in visited_positions:
            reward = -20  # Penalize revisiting a cell

        visited_positions.add(tuple(env.position))  # Add the current position to the set

        agent.remember(state, action, reward, next_state, done)
        loss = agent.replay(batch_size)
        
        if loss != None:
            total_loss += loss

        state = next_state
        total_reward += reward

        done = env.terminated()
        # env.render()

    agent.update_target_network()

    # Print total reward for the episode
    # print(f"Episode {episode + 1}, Total Reward: {total_reward}")
    # print(f"Episode {episode + 1}, Total Loss: {total_loss}")
    loss_every_episode.append(total_loss)
    reward_every_episode.append(total_reward)

    # Save the trained model if needed
    if (episode + 1) % 100 == 0:
        torch.save(agent.q_network.state_dict(), f'model_state_dicts/trained_model_{episode + 1}.pth')

# Save the trained model if needed
torch.save(agent.q_network.state_dict(), 'trained_model.pth')

# Plotting
episodes = list(range(1, num_episodes + 1))  # Create a list from 1 to num_episodes
plt.figure(figsize=(10, 6))
plt.plot(episodes, loss_every_episode, marker='o', linestyle='-')
plt.title('Episode V/s Loss')
plt.xlabel('Episode')
plt.ylabel('Loss')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('loss_plot.jpg')
plt.clf()

plt.figure(figsize=(10, 6))
plt.plot(episodes, reward_every_episode, marker='o', linestyle='-')
plt.title('Episode V/s Reward')
plt.xlabel('Episode')
plt.ylabel('Reward')  # Corrected ylabel to 'Reward'
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('rewards_plot.jpg')
plt.clf()