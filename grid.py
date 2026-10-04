"""Problem generation: grid, obstacles, start and goal -- all from the roll-number seed."""
import random
from collections import deque

ROLL_NUMBER = 80
SEED = ROLL_NUMBER          # seed used for the whole problem instance
GRID_SIZE = 20              # 20 x 20 grid
OBSTACLE_DENSITY = 0.20     # fraction of cells that are blocked


def _reachable(obstacles, size, start, goal):
    """4-connected BFS, only used to make sure the generated problem is solvable."""
    seen, queue = {start}, deque([start])
    while queue:
        x, y = queue.popleft()
        if (x, y) == goal:
            return True
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = (x + dx, y + dy)
            if 0 <= n[0] < size and 0 <= n[1] < size and n not in obstacles and n not in seen:
                seen.add(n)
                queue.append(n)
    return False


def generate_problem(seed=SEED, size=GRID_SIZE, density=OBSTACLE_DENSITY):
    """Return (size, obstacles, start, goal). Nothing is hardcoded: everything comes from random.seed(seed)."""
    random.seed(seed)
    cells = [(x, y) for x in range(size) for y in range(size)]
    while True:
        obstacles = set(random.sample(cells, int(density * size * size)))
        free = [c for c in cells if c not in obstacles]
        start, goal = random.sample(free, 2)
        far_enough = abs(start[0] - goal[0]) + abs(start[1] - goal[1]) >= size  # keep the task non-trivial
        if far_enough and _reachable(obstacles, size, start, goal):
            return size, obstacles, start, goal


if __name__ == "__main__":
    size, obstacles, start, goal = generate_problem()
    print(f"seed={SEED} grid={size}x{size} obstacles={len(obstacles)} start={start} goal={goal}")
