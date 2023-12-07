import numpy as np
import gym
import matplotlib.pyplot as plt

# Q-learning parameters
lr = 0.1          # Learning rate
gamma = 0.99       # Discount factor
epsilon = 0.1      # Epsilon-greedy exploration rate
num_episodes = 10000  # Number of episodes

# Create the FrozenLake environment
env = gym.make('FrozenLake8x8-v1')

# Initialize the Q-table
Q = np.zeros((env.observation_space.n, env.action_space.n))

# Lists to store performance metrics
total_rewards = []
total_steps = []
success_rate = []

# Q-learning algorithm
for episode in range(num_episodes):
    state, info = env.reset()
    done = False
    episode_reward = 0
    episode_steps = 0

    while not done:
        # Epsilon-greedy action selection
        if np.random.rand() < epsilon:
            action = env.action_space.sample()  # Explore
        else:
            action = np.argmax(Q[state])  # Exploit

        next_state, reward, done, _, info = env.step(action)

        # Q-value update
        Q[state, action] = Q[state, action] + lr * \
            (reward + gamma * np.max(Q[next_state]) - Q[state, action])

        state = next_state
        episode_reward += reward
        episode_steps += 1

    # Store episode metrics
    total_rewards.append(episode_reward)
    total_steps.append(episode_steps)
    success_rate.append(int(episode_reward > 0))

    print(
        f"Episode: {episode}, Total Reward: {episode_reward}, Total Steps: {episode_steps}")

# Plotting total rewards
plt.figure(figsize=(10, 5))
plt.plot(total_rewards, label='Total Rewards', color='tab:blue')
plt.title('Total Rewards per Episode')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.legend()
plt.show()

# Plotting total steps
plt.figure(figsize=(10, 5))
plt.plot(total_steps, label='Total Steps', color='tab:orange')
plt.title('Total Steps per Episode')
plt.xlabel('Episode')
plt.ylabel('Total Steps')
plt.legend()
plt.show()

# Print overall metrics
print('----------------------------------------------------------')
print("Overall Average reward:", np.mean(total_rewards))
print("Overall Average number of steps:", np.mean(total_steps))
print("Success rate (%):", np.mean(success_rate) * 100)
