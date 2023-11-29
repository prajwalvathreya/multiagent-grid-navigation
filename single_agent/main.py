import numpy as np
from dqnagent import DQNAgent
from single_agent_environment import MazeEnvironment
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from mazegenerator import *
import json

environments = []
start_positions = []
device = "cpu"

# Create the maze environment
example_maze = np.array([
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

example_start_position = (6,0)
example_goal_position = (12, 15)

env1 = MazeEnvironment(example_maze, example_start_position, example_goal_position, 0)
environments.append(env1)
start_positions.append(example_start_position)

print("Generating environments for training...")
for i in tqdm(range(1, 250)):
    # Create the maze environment
    maze, s_row, s_col, g_row, g_col = generate_maze()
    
    start_position = (s_row, s_col)
    goal_position = (g_row, g_col)

    #generate new environment
    env = MazeEnvironment(maze, start_position, goal_position, i)

    #store environments to train on
    environments.append(env)
    start_positions.append(start_position)
    env.save_environment()

print(len(environments))
print("Completed generating environments!")

# Define DQN parameters
state_size = 16 * 16  # Modify according to your state representation
action_size = 4  # Modify according to the number of discrete actions
agent = DQNAgent(state_size, action_size, device)

action_map = {0: "up", 1: "down", 2: "left", 3: "right"}
loss_for_all_environments = {}
rewards_for_all_environments = {}
env_count = 1

print("Training on environments..")

for i in tqdm(range(len(environments))):

    num_episodes = 1000
    batch_size = 32
    loss_every_episode = []
    reward_every_episode = []
    env = environments[i]
    start_spot = start_positions[i]

    for episode in tqdm(range(num_episodes)):

        state = env.reset()
        total_reward = 0
        total_loss = 0
        done = False

        visited_positions = set()  # Set to store visited positions
        visited_positions.add(start_spot)

        while not env.terminated():

            action = agent.choose_action(state)
            next_state, reward = env.step(action)

            # Check if the next position is already visited
            if tuple(env.position) in visited_positions:
                reward = -50  # Penalize revisiting a cell
            else:
                reward = 10

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
        if (episode + 1) % 500 == 0:
            torch.save(agent.q_network.state_dict(), f'model_state_dicts/{env_count}_model_{episode + 1}.pth')

    loss_for_all_environments[env_count] = loss_every_episode
    rewards_for_all_environments[env_count] = reward_every_episode

    env_count += 1

with open("rewards.json", 'w') as json_file:
    json.dump(rewards_for_all_environments, json_file, indent=2)

with open("loss.json", 'w') as json_file:
    json.dump(loss_for_all_environments, json_file, indent=2)

# # Save the trained model if needed
torch.save(agent.q_network.state_dict(), 'trained_model.pth')