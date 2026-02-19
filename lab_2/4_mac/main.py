from collections import deque

# Check if a state is valid
def is_valid(state):
    M_left, C_left, boat = state
    M_right = 3 - M_left
    C_right = 3 - C_left

    if M_left < 0 or C_left < 0 or M_right < 0 or C_right < 0:
        return False
    if (M_left > 0 and M_left < C_left) or (M_right > 0 and M_right < C_right):
        return False
    return True

# Generate possible moves
def get_moves(state):
    M, C, boat = state
    moves = []
    if boat == 0:  # Boat on left side
        directions = -1
    else:          # Boat on right side
        directions = 1

    # Possible boat moves
    for m in range(3):
        for c in range(3):
            if 1 <= m + c <= 2:  # boat can carry 1 or 2 people
                new_state = (M + directions*m, C + directions*c, 1 - boat)
                if is_valid(new_state):
                    moves.append(new_state)
    return moves

# BFS to find solution
def bfs():
    start = (3, 3, 0)  # All on left side, boat on left
    goal = (0, 0, 1)   # All on right side
    queue = deque([[start]])
    visited = set()
    visited.add(start)

    while queue:
        path = queue.popleft()
        state = path[-1]

        if state == goal:
            return path

        for next_state in get_moves(state):
            if next_state not in visited:
                visited.add(next_state)
                queue.append(path + [next_state])

# Run the program
solution = bfs()
if solution:
    for step, state in enumerate(solution):
        print(f"Step {step}: {state}")
else:
    print("No solution found.")
