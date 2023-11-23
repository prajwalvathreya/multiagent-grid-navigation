import numpy as np
import random

# Define some constants for the maze
START = 10
END = 10
BLOCK = 0
ROUTE_1 = 1
ROUTE_2 = 2
ROUTE_3 = 3

# Define a helper function to check if a cell is valid and not blocked


def is_valid(cell, maze):
    row, col = cell
    return 0 <= row < 16 and 0 <= col < 16

# Define a helper function to get the neighbors of a cell


def get_neighbors(cell, maze):
    row, col = cell
    neighbors = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        new_row = row + dr
        new_col = col + dc
        if is_valid((new_row, new_col), maze):
            neighbors.append((new_row, new_col))
    return neighbors

# Define a helper function to generate a random route from start to end


def generate_route(start, end, maze, route_value):
    # Use a stack to store the cells to visit
    stack = [start]
    # Use a set to store the visited cells
    visited = set()
    # Use a dictionary to store the parent of each cell
    parent = {}
    # Mark the start cell as visited
    visited.add(start)
    # Loop until the stack is empty or the end is found
    while stack and end not in visited:
        # Pop the top cell from the stack
        current = stack.pop()
        # Get the valid neighbors of the current cell
        neighbors = get_neighbors(current, maze)
        # Shuffle the neighbors to add some randomness
        random.shuffle(neighbors)
        # Loop through the neighbors
        for neighbor in neighbors:
            # If the neighbor is not visited, push it to the stack and mark it as visited
            if neighbor not in visited:
                stack.append(neighbor)
                visited.add(neighbor)
                # Store the current cell as the parent of the neighbor
                parent[neighbor] = current
    # If the end is not found, return False
    if end not in visited:
        return False
    # Otherwise, backtrack from the end to the start using the parent dictionary
    current = end
    while current != start:
        # Exclude marking the end cell as part of the route
        if current != end:
            # Mark the current cell as part of the route with the specified route value
            maze[current[0]][current[1]] = route_value
        # Move to the parent of the current cell
        current = parent[current]
    # Return True
    return True


# Define the main function to generate a random maze
def generate_maze():
    # Create an empty 16 * 16 array filled with zeros
    maze = np.zeros((16, 16), dtype=int)
    # Choose a random start position
    start_row = random.randint(0, 15)
    start_col = 0
    start = (start_row, start_col)
    # Choose a random end position
    end_row = random.randint(0, 15)
    end_col = 15
    end = (end_row, end_col)
    # Mark the start and end positions with 10
    maze[start_row][start_col] = START
    maze[end_row][end_col] = END
    # Generate the first route with value 1
    generate_route(start, end, maze, ROUTE_1)
    # # Generate the second route with value 2
    generate_route(start, end, maze, ROUTE_2)
    # # Generate the third route with value 3
    generate_route(start, end, maze, ROUTE_3)
    # Return the maze
    return maze


# Generate the maze with routes 1, 2, 3
# maze = generate_maze()
# print(maze)
