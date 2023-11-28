# SARSA algorithm
import numpy as np

from SarsaAgent import SarsaAgent


def sarsa(agent, env, n_episodes, epsilon=0.1, alpha=0.1, gamma=0.9):
    for episode in range(n_episodes):
        state = env.reset()[0]
        action = epsilon_greedy_policy(agent, state, epsilon)

        while True:
            next_state, reward = env.step([action])
            next_state = next_state[0]
            next_action = epsilon_greedy_policy(agent, next_state, epsilon)

            # SARSA update
            td_error = reward[0] + gamma * agent.get_q_value(
                next_state, next_action) - agent.get_q_value(state, action)
            agent.update_q_value(state, action, alpha * td_error)

            state = next_state
            action = next_action

            env.render()

            print(action)

            if env.is_goal_reached():
                break

        print(episode)

# Epsilon-greedy policy


def epsilon_greedy_policy(agent, state, epsilon):
    if np.random.rand() < epsilon:
        # Explore: choose a random action index
        action_index = np.random.choice(len(agent.get_actions()))
        return agent.get_actions()[action_index]
    else:
        # Exploit: choose the action with the highest Q-value
        return agent.get_best_action(state)
