"""Run PSO path planning on the problem generated from the roll-number seed."""
import numpy as np
from grid import generate_problem, SEED
from planner import make_path, path_length, collided_cells, make_fitness
from pso import PSO
from visualize import plot_path, plot_convergence

N_WAYPOINTS = 8
N_PARTICLES = 150
ITERATIONS = 400


def main():
    size, obstacles, start, goal = generate_problem()
    print(f"Seed (roll number): {SEED} | grid {size}x{size} | obstacles {len(obstacles)} | start {start} | goal {goal}")

    rng = np.random.default_rng(SEED)
    dim = 2 * N_WAYPOINTS
    lower, upper = np.zeros(dim), np.full(dim, float(size))

    fitness = make_fitness(start, goal, obstacles, size)
    # Initialise waypoints at random FREE cell centres (better starting swarm than uniform random)
    free = np.array([(x + 0.5, y + 0.5) for x in range(size) for y in range(size) if (x, y) not in obstacles])
    init = free[rng.integers(0, len(free), (N_PARTICLES, N_WAYPOINTS))].reshape(N_PARTICLES, dim)
    best, best_f, history = PSO(fitness, dim, lower, upper, N_PARTICLES, ITERATIONS,
                                init_positions=init, rng=rng).run()

    path = make_path(best, start, goal)
    hits = collided_cells(path, obstacles, size)
    print(f"Best fitness: {best_f:.3f} | path length: {path_length(path):.3f} | collisions: {len(hits)}")
    plot_path(size, obstacles, start, goal, path, path_length(path), SEED)
    plot_convergence(history)
    print("Saved results/path.png and results/convergence.png")
    return size, obstacles, start, goal, path, history


if __name__ == "__main__":
    main()
