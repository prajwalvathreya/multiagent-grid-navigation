# Function to perform DFS and calculate total reward for an episode
def dfs(agent, env, n_episodes):
    def run_dfs_episode(agent, env):
        total_reward = 0
        visited_states = set()

        def dfs(current_state):
            nonlocal total_reward
            if current_state in visited_states:
                return
            visited_states.add(current_state)

            agent.position = current_state
            total_reward += agent.get_reward()

            if agent.is_goal():
                return

            for action in agent.get_actions():
                agent.do_action(action)
                next_state = agent.position
                dfs(next_state)
                agent.do_action(agent.reverse_action(action))  # Backtrack

        start_state = agent.position
        dfs(start_state)
        return total_reward

    # Run DFS for each episode
    for episode in range(n_episodes):
        total_reward = run_dfs_episode(agent, env)
        print(f"Episode {episode + 1}, Total Reward: {total_reward}")

    # Optionally, you can visualize the final path taken by the agent
    print("Final path:", env.render())
