"""Path encoding, collision checking and fitness function for PSO path planning."""
import numpy as np

STEP = 0.05          # sampling resolution (in cells) when checking a segment for collisions
PENALTY = 30.0       # cost added per collided cell on a path


def make_path(particle, start, goal):
    """A particle is a flat vector of intermediate waypoints (x1,y1,x2,y2,...)."""
    wps = np.asarray(particle).reshape(-1, 2)
    s = (start[0] + 0.5, start[1] + 0.5)      # cell centres
    g = (goal[0] + 0.5, goal[1] + 0.5)
    return np.vstack([s, wps, g])


def path_length(path):
    return float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())


def collided_cells(path, blocked, size):
    """Set of obstacle cells touched by the polyline (sampled along every segment)."""
    hit = set()
    for a, b in zip(path[:-1], path[1:]):
        n = max(2, int(np.linalg.norm(b - a) / STEP))
        for t in np.linspace(0.0, 1.0, n):
            p = a + t * (b - a)
            c = (int(np.floor(p[0])), int(np.floor(p[1])))
            if not (0 <= c[0] < size and 0 <= c[1] < size) or c in blocked:
                hit.add(c)
    return hit


def make_fitness(start, goal, blocked, size):
    def fitness(particle):
        path = make_path(particle, start, goal)
        return path_length(path) + PENALTY * len(collided_cells(path, blocked, size))
    return fitness
