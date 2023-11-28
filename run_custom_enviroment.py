from custom_frozen_lake_env import CustomFrozenLakeEnv


def main():
    # Instantiate the custom environment
    env = CustomFrozenLakeEnv(map_size=4, is_slippery=True)

    # Reset the environment to get the initial state
    initial_state = env.reset()

    # Run a few steps in the environment
    for _ in range(10000000):
        # Sample a random action from the action space
        action = env.action_space.sample()

        # Take a step in the environment
        next_state, reward, done, _ = env.step(action)

        # Print information about the step
        print(
            f"Action: {action}, Next State: {next_state}, Reward: {reward}, Done: {done}")

        # Render the environment (for visualization)
        env.render()

        if done:
            print("Episode finished!")
            break

    # Close the environment when done
    env.close()


if __name__ == "__main__":
    main()
