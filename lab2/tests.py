import numpy as np
from solver import EvolutionaryAlgorithm
import matplotlib.pyplot as plt
import time


def one_run(
    f: callable, dimensions: int, mu: int, sigma: float, max_iter: int, method: str
):
    P0 = np.random.uniform(-100, 100, size=(mu, dimensions))
    solver = EvolutionaryAlgorithm(f, P0, mu, sigma, max_iter=max_iter)
    x_t, f_t = solver.solve(method=method)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].plot(f_t)
    ax[0].set_yscale("log")
    ax[0].set_xlabel("Iteration")
    ax[0].set_ylabel("Objective function value")

    ax[1].plot([x[0] for x in x_t], [x[1] for x in x_t], "-o")
    ax[1].set_xlabel("x")
    ax[1].set_ylabel("y")
    plt.show()


def test_mu(
    f: callable,
    dimensions: int,
    mu_values: list,
    sigma: float,
    max_iter: int,
    method: str,
):

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_yscale("log")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Log(f)")

    for mu in mu_values:

        start = time.time()
        P0 = np.random.uniform(-100, 100, size=(mu, dimensions))
        solver = EvolutionaryAlgorithm(f, P0, mu, sigma, max_iter=max_iter)
        x_t, f_t = solver.solve(method=method)
        ax.plot(f_t, label=f"mu={mu}")
        end = time.time()
        print(f"mu={mu}, time={end-start}")

    ax.legend()
    plt.show()


def test_sigma(
    f: callable,
    dimensions: int,
    mu: int,
    sigma_values: list,
    max_iter: int,
    method: str,
):

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_yscale("log")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Log(f)")

    for sigma in sigma_values:

        start = time.time()
        P0 = np.random.uniform(-100, 100, size=(mu, dimensions))
        solver = EvolutionaryAlgorithm(f, P0, mu, sigma, max_iter=max_iter)
        x_t, f_t = solver.solve(method=method)
        ax.plot(f_t, label=f"sigma={sigma}")
        end = time.time()
        print(f"sigma={sigma}, time={end-start}")

    ax.legend()
    plt.show()


def one_run_mean_stats(
    f: callable,
    dimensions: int,
    mu: int,
    sigma: float,
    max_iter: int,
    method: str,
    k: int,
):

    f_values = []
    time_values = []
    iterations_values = []

    for i in range(k):
        P0 = np.random.uniform(-100, 100, size=(mu, dimensions))
        solver = EvolutionaryAlgorithm(f, P0, mu, sigma, max_iter=max_iter)
        start = time.time()
        x_t, f_t = solver.solve(method=method)
        end = time.time()
        time_values.append(end - start)
        f_values.append(f_t[-1])
        iterations_values.append(len(f_t))

    result = {
        "mean_time": round(np.mean(time_values), 2),
        "std_time": round(np.std(time_values), 2),
        "mean_iterations": round(np.mean(iterations_values), 2),
        "std_iterations": round(np.std(iterations_values), 2),
        "mean_f": round(np.mean(f_values), 2),
        "std_f": round(np.std(f_values), 2),
    }

    return result
