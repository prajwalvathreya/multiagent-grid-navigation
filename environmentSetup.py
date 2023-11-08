import numpy as np

'''
Considering this 2D matrix of 16x16 is our base

In the below maze the start and end states are represented by the value 10.

start = [7, 0]
end = [12, 15]

The numbers 1 represent the path 1.
The number 2 represent the path 2.
The number 3 represent the path 3.

To allow agents to switch paths, I have added overlaps where paths can be switched.
This switch will happen based on the transition probability at that intersection.

Aim :
3 agenst will explore these paths to find the best route. They will also communicate amongst each other so that no agent is in the same cell at the same time. This is done in-order to simulate congestion and how we can avoid it.

'''

start = [7,0]
end = [12, 15]


maze = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0],
    [3, 0, 0, 3, 0, 0, 0, 0, 3, 0, 0, 3, 0, 0, 0, 0],
    [3, 0, 0, 3, 0, 0, 0, 0, 3, 0, 0, 3, 0, 0, 0, 0],
    [3, 0, 0, 3, 0, 0, 0, 0, 3, 0, 0, 3, 0, 0, 0, 0],
    [10, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 3, 0, 0, 0, 0],
    [2, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 3, 0, 0, 0, 0],
    [2, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 3, 0, 0, 0, 0],
    [2, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 3, 0, 0, 0, 0],
    [2, 2, 2, 1, 2, 2, 2, 2, 1, 2, 2, 3, 3, 3, 3, 3],
    [2, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 3],
    [2, 2, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 10],
    [0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
])

print("Given Maze:")
print(maze)

path_1_coords = []
path_2_coords = []
path_3_coords = []

path_1_coords.append(start)
path_2_coords.append(start)
path_3_coords.append(start)

# Loop through the maze array to find path coordinates
for i in range(len(maze)):
    for j in range(len(maze[i])):
        if maze[i][j] == 1:
            path_1_coords.append((i, j))
        elif maze[i][j] == 2:
            path_2_coords.append((i, j))
        elif maze[i][j] == 3:
            path_3_coords.append((i, j))

path_1_coords.append(end)
path_2_coords.append(end)
path_3_coords.append(end)

# Display the path coordinates
print("Path 1 Coordinates:")
print(path_1_coords)
print("\nPath 2 Coordinates:")
print(path_2_coords)
print("\nPath 3 Coordinates:")
print(path_3_coords)

