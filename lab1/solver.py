from autograd import grad
import numpy as np


def solver(f, x0, beta, eps=1e-3, max_iter=1000):
    x = x0
    df = grad(f)
    f_t = [f(x)]
    x_t = [x]
    for i in range(max_iter):
        x = x - beta * df(x)
        f_t.append(f(x))
        x_t.append(x)
        if np.linalg.norm(df(x)) < eps or np.linalg.norm(x_t[-1] - x_t[-2]) < eps:
            break
    return x, f_t
