from tests import one_run, test_mu, one_run_mean_stats, test_sigma
import sys

sys.path.append("WSI/CEC/cec2017-py")
from cec2017.functions import f3, f19

sys.path.append("WSI/lab2")
import pandas as pd

dimensions = 10
mu = 20
sigma = 3
max_iter = 1000
f = f19
method = "tournament"
# one_run(f, dimensions, mu, sigma, max_iter, method)

mu_values = [10, 20, 50, 100]
test_mu(f, dimensions, mu_values, sigma, max_iter, method)

sigma_values = [1, 3, 5, 10]
test_sigma(f, dimensions, mu, sigma_values, max_iter, method)

k = 25
mu_df = pd.DataFrame(
    columns=[
        "mu",
        "średnia czasu",
        "odchylenie czasu",
        "średnia wartości",
        "odchylenie wartości",
        "średnia iteracji",
        "odchylenie iteracji",
    ]
)
for mu in mu_values:
    result = one_run_mean_stats(f, dimensions, mu, sigma, max_iter, method, k)
    mu_df = mu_df._append(
        {
            "mu": mu,
            "średnia czasu": result["mean_time"],
            "odchylenie czasu": result["std_time"],
            "średnia wartości": result["mean_f"],
            "odchylenie wartości": result["std_f"],
            "średnia iteracji": result["mean_iterations"],
            "odchylenie iteracji": result["std_iterations"],
        },
        ignore_index=True,
    )

mu_df.to_csv("mu.csv", index=False)

sigma_df = pd.DataFrame(
    columns=[
        "sigma",
        "średnia czasu",
        "odchylenie czasu",
        "średnia wartości",
        "odchylenie wartości",
        "średnia iteracji",
        "odchylenie iteracji",
    ]
)

for sigma in sigma_values:
    result = one_run_mean_stats(f, dimensions, mu, sigma, max_iter, method, k)
    sigma_df = sigma_df._append(
        {
            "sigma": sigma,
            "średnia czasu": result["mean_time"],
            "odchylenie czasu": result["std_time"],
            "średnia wartości": result["mean_f"],
            "odchylenie wartości": result["std_f"],
            "średnia iteracji": result["mean_iterations"],
            "odchylenie iteracji": result["std_iterations"],
        },
        ignore_index=True,
    )

sigma_df.to_csv("sigma.csv", index=False)
