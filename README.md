
# Multi-Agent Grid Navigation

This project involves training multiple reinforcement learning agents to navigate a maze. The agents learn to move from a start state to a goal state while avoiding collisions with each other.

## Files Description

### `dqn_agent.py`
This file contains the implementation of the DQN (Deep Q-Network) agent. It includes the neural network model used by the agent and methods for choosing actions, remembering experiences, and learning from replayed experiences.

### `maze_environment.py`
Defines the `MazeEnvironment` class, which represents the maze environment where multiple agents navigate. The environment handles the interactions of agents with the maze and each other, updating states, and providing rewards.

### `run.py`
Contains the main training loop where multiple agents are trained in the maze environment. This script initializes the agents and the environment, and iterates through training episodes, allowing agents to learn from their experiences.

### `evaluation.py`
Provides functionality to evaluate the performance of the agents. It runs the agents through a series of episodes in the maze environment and calculates their average reward as a measure of performance.

### `value_iteration_perf.py`
Provides the optimal policy graph and convergence of the value iteration for a pre-defined environment (8x8 grid).

### `value_iteration_predefined_env.py`
Provides the Vopt values of each state in a 8x8 grid where it provides information on how the Vopt value changes as we get closer to the goal state. It also provides an insight on 
how the rewards are calculated for a given number of iterations using value iteration. 

## Requirements
To run the project, ensure the following libraries are installed:
- numpy
- torch
- pygame

You can install these using the `requirements.txt` file with the command `pip install -r requirements.txt`.

## Running the Project
Execute `run.py` to start the training process. The training progress can be observed through the console output and the Pygame window visualizing the agents' movements in the maze.

Execute `value_iteration_perf.py` and `value_iteration_predefined_env.py` to view the performance of value iteration for a pre-defined environment. 
