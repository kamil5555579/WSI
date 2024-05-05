import numpy as np
import matplotlib.pyplot as plt
from solver import solver

def q(x, alpha=1):
    n = len(x)
    sum_value = 0
    for i in range(1, n + 1):
        sum_value += alpha ** ((i - 1) / (n - 1)) * x[i - 1] ** 2
    return sum_value

def test_on_function(f, beta):
    n = 10
    x = np.random.uniform(-100, 100, size=n)
    for beta in beta:
        x_opt, f_t = solver(f, x, beta, max_iter=1000)
        print("Optimal x:", x_opt)
        print("Optimal value:", f(x_opt))
        plt.plot(f_t, label=f"beta={beta}")
    plt.legend()
    plt.xlabel("Iteracja")
    plt.yscale("log")
    plt.ylabel("Logarytm wartości funkcji celu")
    plt.savefig("test_on_function.png")
    plt.close()

def test_alpha(alpha, beta):
    n = 10
    x = np.random.uniform(-100, 100, size=n)
    q_alpha = lambda x: q(x, alpha)
    for beta in beta:
        x_opt, f_t = solver(q_alpha, x, beta, max_iter=1000)
        print("Optimal x:", x_opt)
        print("Optimal value:", q_alpha(x_opt))
        plt.plot(f_t, "-o", label=f"beta={beta}")
    plt.legend()
    plt.title(f"alpha={alpha}")
    plt.xlabel("Iteracja")
    plt.ylabel("Wartość funkcji celu")
    plt.savefig("test_alpha" + str(alpha) + ".png")
    plt.close()


def test_alpha_beta(alpha, beta, k=10):
    n = 10
    iters = []
    optimal_values = []
    for i in range(k):
        x = np.random.uniform(-100, 100, size=n)
        q_alpha = lambda x: q(x, alpha)
        x_opt, f_t = solver(q_alpha, x, beta, max_iter=1000)
        iters.append(len(f_t))
        optimal_values.append(q_alpha(x_opt))

    return (
        np.mean(iters),
        np.mean(optimal_values),
        np.std(iters),
        np.std(optimal_values),
    )
