"""Matplotlib visualisation of the grid, obstacles, start, goal, best path and convergence."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def plot_path(size, obstacles, start, goal, path, length, seed, out="results/path.png"):
    fig, ax = plt.subplots(figsize=(7, 7))
    for (x, y) in obstacles:
        ax.add_patch(Rectangle((x, y), 1, 1, color="#333333"))
    ax.add_patch(Rectangle(start, 1, 1, color="#2ca02c", alpha=0.8))
    ax.add_patch(Rectangle(goal, 1, 1, color="#d62728", alpha=0.8))
    ax.plot(path[:, 0], path[:, 1], "-o", color="#1f77b4", lw=2, ms=4, label=f"PSO path (length {length:.2f})")
    ax.plot([], [], "s", color="#2ca02c", label="Start")
    ax.plot([], [], "s", color="#d62728", label="Goal")
    ax.plot([], [], "s", color="#333333", label="Obstacle")
    ax.set_xlim(0, size); ax.set_ylim(0, size); ax.set_aspect("equal")
    ax.set_xticks(range(size + 1)); ax.set_yticks(range(size + 1))
    ax.tick_params(labelsize=7); ax.grid(True, lw=0.3)
    ax.set_title(f"PSO path planning (seed = {seed})")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.04), ncol=4, fontsize=8)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_convergence(history, out="results/convergence.png"):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(history, color="#ff7f0e")
    ax.set_xlabel("Iteration"); ax.set_ylabel("Best fitness (length + penalty)")
    ax.set_title("PSO convergence"); ax.grid(True, lw=0.3)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
