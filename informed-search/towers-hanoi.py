#!/usr/bin/env python3

import heapq

class HanoiState:
    def __init__(self, towers, parent=None, move=None, depth=0, cost=0):
        """
        Initialize a state in the Towers of Hanoi problem.

        :param towers: A tuple of tuples representing the disks on each rod.
        :param parent: The parent state from which this state was derived.
        :param move: The move taken to reach this state (disk, from_rod, to_rod).
        :param depth: The depth of this state in the search tree (g(n)).
        :param cost: The total estimated cost of this state (f(n) = g(n) + h(n)).
        """
        self.towers = towers
        self.parent = parent
        self.move = move
        self.depth = depth
        self.cost = cost

    def __lt__(self, other):
        """Define less-than for priority queue based on cost."""
        return self.cost < other.cost

    def __hash__(self):
        """Allow the state to be used in sets and dictionaries."""
        return hash(self.towers)

    def __eq__(self, other):
        """Check equality based on the towers' configuration."""
        return self.towers == other.towers

def heuristic(state, goal_tower_index):
    """
    Estimate the cost to reach the goal state.

    The heuristic counts the number of disks not yet in the correct position
    on the goal rod.

    :param state: The current HanoiState.
    :param goal_tower_index: The index of the goal rod (0-based).
    :return: Estimated cost to reach the goal from the current state.
    """
    count = 0
    goal_tower = state.towers[goal_tower_index]
    # Count disks on the goal rod that are not in the correct order
    for i in range(len(goal_tower)):
        expected_disk = num_disks - i
        if goal_tower[i] != expected_disk:
            count += 1
    # Add disks not yet on the goal rod
    for i, tower in enumerate(state.towers):
        if i != goal_tower_index:
            count += len(tower)
    return count

def get_neighbors(state, num_rods):
    """
    Generate all valid neighboring states from the current state.

    :param state: The current HanoiState.
    :param num_rods: The total number of rods in the puzzle.
    :return: A list of neighboring HanoiState instances.
    """
    neighbors = []
    # Iterate over all rods
    for i in range(num_rods):
        if state.towers[i]:
            # Get the top disk from rod i
            disk = state.towers[i][-1]
            # Attempt to move the disk to every other rod
            for j in range(num_rods):
                if i != j:
                    # Check if the move is valid
                    if not state.towers[j] or state.towers[j][-1] > disk:
                        # Create a deep copy of the towers to simulate the move
                        new_towers = list(list(tower) for tower in state.towers)
                        # Move the disk from rod i to rod j
                        new_towers[j].append(new_towers[i].pop())
                        # Create a new HanoiState with the updated towers
                        new_state = HanoiState(
                            towers=tuple(tuple(tower) for tower in new_towers),
                            parent=state,
                            move=(disk, i, j),
                            depth=state.depth + 1
                        )
                        neighbors.append(new_state)
    return neighbors

def astar(start_state, goal_tower_index):
    """
    Perform the A* search algorithm to solve the Towers of Hanoi problem.

    :param start_state: The initial HanoiState.
    :param goal_tower_index: The index of the goal rod (0-based).
    :return: The goal HanoiState if a solution is found, else None.
    """
    open_set = []
    # Initialize the priority queue with the start state
    heapq.heappush(open_set, (start_state.cost, start_state))
    closed_set = set()  # Keep track of visited states to avoid loops
    while open_set:
        # Pop the state with the lowest estimated total cost (f(n))
        _, current_state = heapq.heappop(open_set)
        # Check if the goal state is reached
        if is_goal(current_state, goal_tower_index):
            return current_state
        closed_set.add(current_state)
        # Generate all valid neighboring states
        for neighbor in get_neighbors(current_state, num_rods=3):
            if neighbor in closed_set:
                continue
            # Calculate the actual cost to reach this neighbor (g(n))
            g = neighbor.depth
            # Estimate the cost from the neighbor to the goal (h(n))
            h = heuristic(neighbor, goal_tower_index)
            neighbor.cost = g + h  # Total estimated cost (f(n))
            # Add the neighbor to the open set for exploration
            heapq.heappush(open_set, (neighbor.cost, neighbor))
    return None  # No solution found

def is_goal(state, goal_tower_index):
    """
    Determine if the current state is the goal state.

    The goal state is achieved when all disks are on the goal rod in the correct order.

    :param state: The current HanoiState.
    :param goal_tower_index: The index of the goal rod (0-based).
    :return: True if the goal state is reached, False otherwise.
    """
    goal_tower = state.towers[goal_tower_index]
    # Check if the goal rod contains all disks in the correct order
    return len(goal_tower) == num_disks and all(
        disk == num_disks - i for i, disk in enumerate(goal_tower)
    )

def print_solution(state):
    """
    Backtrack from the goal state to reconstruct and print the solution path.

    :param state: The goal HanoiState.
    """
    path = []
    states = []
    # Traverse back to the start state
    while state.parent is not None:
        path.append(state.move)
        states.append(state)
        state = state.parent
    states.append(state)  # Include the initial state
    path.reverse()        # Reverse the path to get the correct order
    states.reverse()
    print("Solution found in {} moves:".format(len(path)))
    for idx, state in enumerate(states):
        print("\nMove {}: {}".format(idx, f"Move disk {path[idx-1][0]} from rod {path[idx-1][1]+1} to rod {path[idx-1][2]+1}" if idx > 0 else "Initial State"))
        print_towers(state.towers)

def print_towers(towers):
    """
    Print the current state of the towers.

    :param towers: A tuple of tuples representing the disks on each rod.
    """
    max_height = max(len(tower) for tower in towers)
    for level in range(max_height, 0, -1):
        line = ''
        for tower in towers:
            if len(tower) >= level:
                line += ' ' + str(tower[level - 1]) + ' '
            else:
                line += ' | '
        print(line)
    print('---' * len(towers))

def main():
    """
    Main function to execute the Towers of Hanoi solver.
    """
    print("Towers of Hanoi Solver using A* Search")
    global num_disks
    try:
        # Prompt the user to enter the number of disks
        num_disks = int(input("Enter the number of disks: "))
        if num_disks <= 0:
            raise ValueError
    except ValueError:
        print("Invalid input. Using default of 3 disks.")
        num_disks = 3
    # Initialize the starting towers with all disks on the first rod
    start_towers = (tuple(range(num_disks, 0, -1)), (), ())
    goal_tower_index = 2  # Define the goal rod (third rod)
    # Create the initial state with the heuristic cost
    start_state = HanoiState(
        towers=start_towers,
        depth=0,
        cost=heuristic(HanoiState(towers=start_towers), goal_tower_index)
    )
    # Execute the A* search algorithm
    result = astar(start_state, goal_tower_index)
    if result:
        # If a solution is found, print the sequence of moves
        print_solution(result)
    else:
        print("No solution found.")

if __name__ == '__main__':
    main()
