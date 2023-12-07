import numpy as np
import matplotlib.pyplot as plt
import gym

def value_iteration(env, gamma=0.99, theta=1e-8):
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    V = np.zeros(num_states)
    delta_values = []  # to store the maximum change in V over iterations

    while True:
        delta = 0
        for s in range(num_states):
            v = V[s]
            bellman_backup = [sum([p * (r + gamma * V[s_]) for p, s_, r, _ in env.P[s][a]]) for a in range(num_actions)]
            V[s] = max(bellman_backup)
            delta = max(delta, abs(v - V[s]))

        delta_values.append(delta)

        if delta < theta:
            break

    return V, delta_values

def plot_convergence(delta_values):
    print(len(delta_values))
    plt.plot(delta_values)
    plt.xlabel('Iteration')
    plt.ylabel('Delta (Change in V)')
    plt.title('Value Iteration Convergence')
    plt.show()

def plot_policy(env, V, gamma = 0.99):
    policy = np.zeros(env.observation_space.n, dtype=int)
    for s in range(env.observation_space.n):
        bellman_backup = [sum([p * (r + gamma * V[s_]) for p, s_, r, _ in env.P[s][a]]) for a in range(env.action_space.n)]
        policy[s] = np.argmax(bellman_backup)

    plt.imshow(policy.reshape((int(np.sqrt(env.observation_space.n)), -1)), cmap='gray', interpolation='none', origin='upper')
    plt.colorbar()
    plt.title('Optimal Policy')
    plt.show()

if __name__ == "__main__":
    env = gym.make("FrozenLake8x8-v1")
    optimal_values, delta_values = value_iteration(env)
    plot_policy(env, optimal_values)
    plot_convergence(delta_values)
