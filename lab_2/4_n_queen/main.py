class NQueen:
    def __init__(self, n):
        self.n = n
        self.initial_state = ()

    # Goal test
    def goalTest(self, state):
        return len(state) == self.n

    # Check if placing queen is safe
    def is_safe(self, state, col):
        row = len(state)
        for r, c in enumerate(state):
            if c == col or abs(c - col) == abs(r - row):
                return False
        return True

    # Successor function
    def successor(self, state):
        children = []
        for col in range(self.n):
            if self.is_safe(state, col):
                children.append(state + (col,))
        return children

    # DFS Search
    def dfs(self):
        OPEN = [self.initial_state]
        CLOSED = {self.initial_state: None}

        while OPEN:
            current = OPEN.pop()

            if self.goalTest(current):
                return self.generate_path(CLOSED, current)

            for child in self.successor(current):
                if child not in CLOSED:
                    OPEN.append(child)
                    CLOSED[child] = current # type: ignore

        return None

    # Generate solution path
    def generate_path(self, CLOSED, goal_state):
        path = []
        while goal_state is not None:
            path.append(goal_state)
            goal_state = CLOSED[goal_state]
        return path[::-1]

    # Display board
    def display(self, solution):
        for row in solution:
            print(row)
        print("\nBoard Representation:")
        for r in range(self.n):
            line = ""
            for c in range(self.n):
                line += "Q " if solution[r] == c else ". "
            print(line)


# Driver Code
n = 4
agent = NQueen(n)

solution_path = agent.dfs()

print("Solution Path:")
for state in solution_path: # type: ignore
    print(state)

print("\nFinal Solution:")
agent.display(solution_path[-1]) # type: ignore
