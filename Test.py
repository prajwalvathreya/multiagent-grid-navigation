import random
import numpy as np
from tqdm import tqdm
import gymnasium as gym
import matrix_mdp
import sys
import matplotlib.pyplot as plt

nS = 16
nA = 4

slip_prob = 0.1

actions = ['up', 'down', 'left', 'right']

p_0 = np.array([0 for _ in range(nS)])
p_0[12] = 1

P = np.zeros((nS, nS, nA), dtype=float)


def valid_neighbors(i, j):
    neighbors = {}
    if i > 0:
        neighbors[0] = (i-1, j)
    if i < 3:
        neighbors[1] = (i+1, j)
    if j > 0:
        neighbors[2] = (i, j-1)
    if j < 3:
        neighbors[3] = (i, j+1)
    return neighbors


for i in range(4):
    for j in range(4):
        if i == 0 and j == 2:
            continue
        if i == 3 and j == 1:
            continue

        neighbors = valid_neighbors(i, j)
        for a in range(nA):
            if a in neighbors:
                P[neighbors[a][0]*4+neighbors[a][1], i*4+j, a] = 1-slip_prob
                for b in neighbors:
                    if b != a:
                        P[neighbors[b][0]*4+neighbors[b][1], i*4+j,
                            a] = slip_prob/float(len(neighbors.items())-1)

R = np.zeros((nS, nS, nA))

R[2, 1, 3] = 2000
R[2, 3, 2] = 2000
R[2, 6, 0] = 2000

R[13, 9, 1] = 2
R[13, 14, 2] = 2
R[13, 12, 3] = 2

R[11, 15, 0] = -100
R[11, 7, 1] = -100
R[11, 10, 3] = -100
R[10, 14, 0] = -100
R[10, 6, 1] = -100
R[10, 11, 2] = -100
R[10, 9, 3] = -100
R[9, 10, 2] = -100
R[9, 13, 0] = -100
R[9, 5, 1] = -100
R[9, 8, 3] = -100

env = gym.make('matrix_mdp/MatrixMDP-v0', p_0=p_0,
               p=P, r=R, disable_env_checker=True)


#################################################################
# Helper Functions
#################################################################

# reverse map observations in 0-15 to (i,j)
def reverse_map(observation):
    return observation//4, observation % 4

#################################################################
# Q-Learning
#################################################################


'''
In this section, you will implement Q-learning with epsilon-greedy exploration.
The Q-table is initialized to all zeros. The value of eta should be updated as 1/(1 + number of updates) inside the loop.
The value of epsilon should be decayed to (0.9999 * epsilon) each time inside the loop.

Refer to the written assignment for the update equation. Similar to MDPs, use the following code to take an action:

observation, reward, terminated, truncated, info = env.step(action)

Unlike MDPs, your action is now chosen by the epsilon-greedy policy. The action is chosen as follows:

With probability epsilon, choose a random action.
With probability (1 - epsilon), choose the action that maximizes the Q-value (based on the last estimate). 
In case of ties, choose the action with the smallest index.
In case the chosen action is not a legal move, generate a random legal action.

The episode terminates when the agent reaches one of many terminal states. 

After 10, 100, 1000 and 10000 updates, plot a heatmap of V_opt(s) for all states s. This should be a 4x4 grid, corresponding to our map of Mordor.
Please use plt.savefig() to save the plot, and do not use plt.show().
Add each heatmap (clearly labeled) to your answer to Q9 in the written submission.

'''

# Q-learning parameters
Q = np.zeros((nS, nA))
gamma = 0.9
eta = 1
epsilon = 0.9  # Initialize epsilon

# Loop for a certain number of episodes (e.g. 10000)
for episode in tqdm(range(10000)):

    # Reset the environment and get the initial observation (state)
    observation, info = env.reset()
    print(observation)

    # Loop until the episode terminates
    terminated = False
    while not terminated:

        # -------------------------------------- CITATION ----------------------------------------
        # Used Chat GPT here to understand how we are choosing the action based on the given rules.
        # Choose an action using epsilon-greedy policy
        if np.random.random() < epsilon:  # With probability epsilon, choose a random action
            action = random.randint(0, nA - 1)
        else:  # With probability 1-epsilon, choose the action that maximizes the Q-value
            max_q_value = np.max(Q[observation])
            # Find all actions with the same Q-value as the maximum
            max_actions = [a for a in range(
                nA) if Q[observation, a] == max_q_value]
            # Choose the action with the smallest index
            action = min(max_actions)

        # Ensure the chosen action is legal from the current state
        if P[:, observation, action].sum() > 0:
            # Take the action and observe the next state, reward, and termination flag
            next_observation, reward, terminated, truncated, info = env.step(
                action)

            if reward is None:
                reward = 0

            # Q-learning update
            Q[observation, action] = (
                1 - eta) * Q[observation, action] + eta * (reward + gamma * np.max(Q[next_observation]))

            # Set the current state to the next state
            observation = next_observation
        else:
            continue

    # Decay epsilon at the end of each episode
    epsilon = 0.9999 * epsilon

    # Initialize eta for this episode
    eta = 1.0 / (1 + episode)

    # After 10, 100, 1000, and 10000 episodes, generate and save heatmaps
    if episode + 1 in [10, 100, 1000, 10000]:
        # V_opt(s) is the maximum Q-value for each state s
        V_opt = np.max(Q, axis=1)
        plt.figure()
        plt.title(f'Heatmap of V_opt(s) after {episode + 1} episodes')
        plt.imshow(V_opt.reshape(4, 4), cmap='hot')
        plt.colorbar()
        plt.savefig(f'heatmap_{episode + 1}.png')  # Save the plot

# Save the final Q-table
np.savetxt('qtable.txt', Q, fmt='%f')

# Print policy
policy = np.array([actions[i] for i in np.argmax(Q, axis=1)])
print(policy.reshape(4, 4))
