import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque
import numpy as np

# Define the Deep Q-Network (DQN) class


class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()

        # Unpack the state_size tuple
        input_size = np.prod(state_size)

        # Define neural network layers
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten input
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Define the Deep Q-Network Agent class


class DQNAgent:
    def __init__(self, state_size, action_size, device, learning_rate=1e-4, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.0005, memory_size=2000):
        # Initialize Q-network, target network, optimizer, and other parameters
        self.q_network = DQN(state_size, action_size).to(device)
        self.target_network = DQN(state_size, action_size).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(
            self.q_network.parameters(), lr=learning_rate)
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = epsilon
        self.gamma = gamma
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.memory = deque(maxlen=memory_size)
        self.device = device

    def choose_action(self, state):
        # Epsilon-greedy action selection for exploration and exploitation
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(range(self.action_size))  # Explore
        else:
            with torch.no_grad():
                state = torch.tensor(state, dtype=torch.float32).view(
                    1, -1).to(self.device)  # Flatten the input
                q_values = self.q_network(state)
                return torch.argmax(q_values).item()  # Exploit

    def train(self, experiences):
        # Training the Q-network using a batch of experiences
        states, actions, rewards, next_states, dones = experiences
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        q_values = self.q_network(states)
        next_q_values = self.target_network(next_states)
        target_q_values = q_values.clone()

        self.optimizer.zero_grad()

        for i in range(len(q_values)):
            if dones[i]:
                target_q_values[i][actions[i]] = rewards[i]
            else:
                target_q_values[i][actions[i]] = rewards[i] + \
                    self.gamma * torch.max(next_q_values[i])

        loss = nn.MSELoss()(q_values, target_q_values)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        # Update the target network weights with the Q-network weights
        self.target_network.load_state_dict(self.q_network.state_dict())

    def remember(self, state, action, reward, next_state, done):
        # Store an experience tuple in the replay memory
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        # Experience replay to train the Q-network
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)

        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = np.array(states)
        next_states = np.array(next_states)

        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(
            next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.bool).to(self.device)

        experiences = (states, actions, rewards, next_states, dones)
        loss = self.train(experiences)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss

    def reset_weights(self):
        # Reset the weights of the Q-network
        self.q_network.apply(self.initialize_weights)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(
            self.q_network.parameters(), lr=self.learning_rate)

    def initialize_weights(self, layer):
        # Initialize weights of the neural network using Xavier initialization
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(
                layer.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(layer.bias, 0)

    def save_model(self, filename):
        # Save the Q-network model to a file
        torch.save(self.q_network.state_dict(), filename)

    def load_model(self, filename):
        # Load the Q-network model from a file
        self.q_network.load_state_dict(torch.load(filename))
        self.target_network.load_state_dict(self.q_network.state_dict())
