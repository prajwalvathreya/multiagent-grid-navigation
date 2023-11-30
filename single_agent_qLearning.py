# Importing Libraries
import random
import numpy as np
import gym
import matplotlib.pyplot as plt

# Set the learning rate, discount factor, and number of episodes (hardcoded values)
lr = 1
gamma = 0.9
num_episodes = 10000

# Epsilon parameters
epsilon = 0.9
epsilon_decay = 0.999  # You can adjust the decay rate

# Create the environment
env = gym.make('FrozenLake-v1')

# Initialize the Q-table
Q = np.random.uniform(low=0, high=0.01, size=(
    env.observation_space.n, env.action_space.n))

# Lists to store performance metrics
total_rewards = []
total_steps = []
success_rate = []
test_reward = []

# Run the Q-learning algorithm
for i in range(num_episodes):
    s, info = env.reset()
    done = False
    episode_reward = 0
    num_steps = 0

    while not done:
        # Epsilon-greedy action selection
        if np.random.random() < epsilon:  # With probability epsilon, choose a random action
            action = random.randint(0, 3)
        else:  # With probability 1-epsilon, choose the action that maximizes the Q-value
            max_q_value = np.max(Q[s])
            # Find all actions with the same Q-value as the maximum
            max_actions = [a for a in range(
                4) if Q[s, a] == max_q_value]
            # Choose the action with the smallest index
            action = min(max_actions)

        s_, r, done, _, info = env.step(action)

        # Update Q values
        Q[s, action] = Q[s, action] + lr * \
            (r + gamma * np.max(Q[s_, :]) - Q[s, action])
        s = s_
        num_steps += 1
        episode_reward += r

    # Decay epsilon
    epsilon = epsilon * epsilon_decay
    lr = 1.0 / (1 + i)

    # Append performance metrics to lists
    total_rewards.append(episode_reward)
    total_steps.append(num_steps)
    success_rate.append(int(episode_reward > 0))

    # Print episode metrics
    print("Episode:", i + 1, "Reward:", episode_reward, "Steps:", num_steps)

plt.plot(total_steps, 'tab:purple')
plt.title('Number of Steps per Episode')
plt.xlabel('Episode')
plt.ylabel('Number of Steps')

plt.show()

# Calculate the sigmoid-transformed rewards
reward_sigmoid = 1 / (1 + np.exp(-np.array(total_rewards)))

# Plotting
# Calculate the sigmoid-transformed rewards
reward_sigmoid = 1 / (1 + np.exp(-np.array(total_rewards)))

# Plotting
plt.plot(range(1, num_episodes + 1), reward_sigmoid, 'tab:green')
plt.title('Reward as Sigmoid')
plt.xlabel('Episode')
plt.ylabel('Sigmoid(Reward)')

plt.show()

plt.show()

# Print overall metrics
print('----------------------------------------------------------')
print("Overall Average reward:", np.mean(total_rewards))
print("Overall Average number of steps:", np.mean(total_steps))
print("Success rate (%):", np.mean(success_rate) * 100)
