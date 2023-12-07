# SARSA algorithm
import numpy as np
from SarsaAgent import SarsaAgent  # Assuming SarsaAgent is a defined class


def sarsa(agent, env, n_episodes, epsilon=0.1, alpha=0.1, gamma=0.9):
    total_rewards = []  # List to store the total rewards for each episode
    for episode in range(n_episodes):
        env.reset()
        state = env.get_agent_state(0)
        action = epsilon_greedy_policy(agent, state, epsilon)
        episode_reward = 0  # Initialize the total reward for the current episode

        while True:
            next_state, reward = env.step([action])
            next_action = epsilon_greedy_policy(agent, next_state, epsilon)

            # SARSA update
            td_error = reward[0] + gamma * agent.get_q_value(
                next_state, next_action) - agent.get_q_value(state, action)
            agent.update_q_value(state, action, alpha * td_error)

            state = next_state
            action = next_action
            # Accumulate the reward for the current episode
            episode_reward += reward[0]

            env.render()

            print(action)

            if env.is_goal_reached():
                break

        total_rewards.append(episode_reward)

    return total_rewards

# Epsilon-greedy policy


def epsilon_greedy_policy(agent, state, epsilon):
    if np.random.rand() < epsilon:
        # Explore: choose a random action index
        action_index = np.random.choice(len(agent.get_actions()))
        return agent.get_actions()[action_index]
    else:
        # Exploit: choose the action with the highest Q-value
        return agent.get_best_action(state)
