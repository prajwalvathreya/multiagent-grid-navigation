import numpy as np

def evaluate_agent(agent, maze_env, num_episodes):

    total_rewards = []
    
    for episode in range(num_episodes):
        state = maze_env.reset()  # Initialize the environment
        total_reward = 0
        done = False

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = maze_env.step(action)
            total_reward += reward
            state = next_state

        total_rewards.append(total_reward)

    average_reward = np.mean(total_rewards)
    return average_reward
