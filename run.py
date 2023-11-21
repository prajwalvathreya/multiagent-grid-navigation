import numpy as np
import torch
from maze_environment import MazeEnvironment
from dqn_agent import DQNAgent
from evaluation import evaluate_agent

# Define the maze
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

# Initialize the environment and the agent
env = MazeEnvironment(maze=maze, start_positions=[[6, 0], [6, 0]], goal_position=[12, 15])
agent = DQNAgent(state_size=(16, 16), action_size=4, epsilon=1.0, gamma=0.95)


# Set parameters for training
n_episodes = 1000
max_timesteps_per_episode = 100
batch_size = 32

save_interval = 50  # Save every 50 episodes
eval_interval = 10  # Evaluate every 10 episodes

for e in range(n_episodes):
    states = env.reset()
    total_reward = 0
    
    for t in range(max_timesteps_per_episode):
        actions = [agent.act(np.array(state).reshape((16, 16))) for state in states]
        next_states, rewards = env.step(actions)
        dones = [False] * len(states)  # Modify based on your environment logic

        for i, state in enumerate(states):
            agent.remember(state, actions[i], rewards[i], next_states[i], dones[i])
            total_reward += rewards[i]

        states = next_states

        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

        env.render()

        if all(dones):
            break

    print(f"Episode {e+1}/{n_episodes}, Total Reward: {total_reward}")

    if (e + 1) % save_interval == 0:
        torch.save(agent.model.state_dict(), f'model_state_dicts/dqn_model_episode_{e+1}.pth')

    if (e + 1) % eval_interval == 0:
        avg_reward = evaluate_agent(env, agent)
        print(f"Episode {e+1}, Evaluation Average Reward: {avg_reward}")

    agent.epsilon = max(agent.epsilon_min, agent.epsilon_decay * agent.epsilon)

# Save the final model
torch.save(agent.model.state_dict(), 'dqn_model_final.pth')
