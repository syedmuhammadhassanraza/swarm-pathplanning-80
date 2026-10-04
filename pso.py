"""Generic PSO optimiser (minimisation) with linearly decaying inertia and velocity clamping."""
import numpy as np


class PSO:
    def __init__(self, fitness, dim, lower, upper, n_particles=80, iterations=300,
                 w_max=0.9, w_min=0.4, c1=1.8, c2=1.8, init_positions=None, rng=None):
        self.fitness, self.dim = fitness, dim
        self.lower, self.upper = np.asarray(lower, float), np.asarray(upper, float)
        self.n, self.iters = n_particles, iterations
        self.w_max, self.w_min, self.c1, self.c2 = w_max, w_min, c1, c2
        self.rng = rng or np.random.default_rng()
        self.vmax = 0.2 * (self.upper - self.lower)
        self.init_positions = init_positions

    def run(self):
        rng = self.rng
        # 1. initialise positions and velocities
        x = self.init_positions.copy() if self.init_positions is not None else \
            rng.uniform(self.lower, self.upper, (self.n, self.dim))
        v = rng.uniform(-self.vmax, self.vmax, (self.n, self.dim))
        # 2. evaluate, set personal and global bests
        f = np.array([self.fitness(p) for p in x])
        pbest, pbest_f = x.copy(), f.copy()
        g = pbest_f.argmin()
        gbest, gbest_f = pbest[g].copy(), pbest_f[g]
        history = [gbest_f]
        for t in range(self.iters):
            w = self.w_max - (self.w_max - self.w_min) * t / max(1, self.iters - 1)
            r1, r2 = rng.random((self.n, self.dim)), rng.random((self.n, self.dim))
            # 3. velocity and position update
            v = w * v + self.c1 * r1 * (pbest - x) + self.c2 * r2 * (gbest - x)
            v = np.clip(v, -self.vmax, self.vmax)
            x = np.clip(x + v, self.lower, self.upper)
            # 4. evaluate and update bests
            f = np.array([self.fitness(p) for p in x])
            better = f < pbest_f
            pbest[better], pbest_f[better] = x[better], f[better]
            g = pbest_f.argmin()
            if pbest_f[g] < gbest_f:
                gbest, gbest_f = pbest[g].copy(), pbest_f[g]
            history.append(gbest_f)
        return gbest, gbest_f, history
