from SingleAgent import *
from generate_maze import *
from maze_environment import *

# Function to perform DFS and calculate total reward for an episode


def dfs(agent, env, n_episodes):
    def run_dfs_episode(agent, env):
        # Generate a maze and initialize the agent
        maze, start_row, start_col, end_row, end_col = generate_maze_single_agent()
        reward = {0: -100, 1: 0, 2: 0, 3: 0, 10: 100}
        agent = SingleAgent((start_row, start_col), maze, reward)
        env = MazeEnvironment(maze=maze, start_positions=[
                              [start_row, start_col]], goal_position=[end_row, end_col])

        total_reward = 0
        visited_states = set()

        def dfs(current_state):
            nonlocal total_reward
            # Avoid revisiting the same state
            if current_state in visited_states:
                return
            visited_states.add(current_state)

            # Set agent's position to the current state
            agent.position = current_state
            total_reward += agent.get_reward()

            if agent.is_goal():
                return

            # Explore possible actions and move to the next state
            for action in agent.get_actions():
                agent.do_action(action)
                next_state = agent.position
                dfs(next_state)
                # Backtrack to previous state
                agent.do_action(agent.reverse_action(action))

        start_state = agent.position
        dfs(start_state)
        return total_reward

    # Run DFS for each episode
    for episode in range(n_episodes):
        total_reward = run_dfs_episode(agent, env)
        print(f"Episode {episode + 1}, Total Reward: {total_reward}")

    # Optionally, you can visualize the final path taken by the agent
    print("Final path:", env.render())
