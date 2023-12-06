import numpy as np


def evaluate_agent(agent, maze_env, num_episodes):
    # Initialize a list to store total rewards obtained in each episode
    total_rewards = []

    # Iterate over the specified number of episodes
    for episode in range(num_episodes):
        state = maze_env.reset()  # Initialize the environment
        total_reward = 0
        done = False

        # Continue the episode until the environment signals it is done
        while not done:
            # Choose an action using the agent's policy
            action = agent.choose_action(state)
            # Take the chosen action and get the next state and rewards
            next_states, rewards = maze_env.step([action])
            total_reward += rewards  # Accumulate the total reward obtained in the episode
            state = next_states[0] if isinstance(
                next_states, list) else next_states  # Update the current state

        # Store the total reward for the current episode
        total_rewards.append(total_reward)

    # Calculate the average reward over all episodes
    average_reward = np.mean(total_rewards)
    return average_reward
