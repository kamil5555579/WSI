from tests import test_alpha, test_alpha_beta, test_on_function
import pandas as pd
import sys
sys.path.append('WSI/CEC/cec2017-py')
from cec2017.functions import f3, f19
sys.path.append('WSI/lab1')
import numpy as np

# test_alpha(alpha=1, beta=[0.1, 0.3, 0.5])
# test_alpha(alpha=10, beta=[0.05, 0.08])
# test_alpha(alpha=100, beta=[0.005, 0.008])

# alpha_1 = pd.DataFrame(
#     columns=[
#         "beta",
#         "średnia iteracji",
#         "średnia optymalnej wartości",
#         "odchylenie iteracji ",
#         "odchylenie optymalnej wartości",
#     ]
# )
# for beta in [0.1, 0.3, 0.5]:
#     mean_iters, mean_optimal_values, std_iters, std_optimal_values = test_alpha_beta(
#         alpha=1, beta=beta, k=25
#     )
#     alpha_1 = alpha_1._append(
#         {
#             "beta": beta,
#             "średnia iteracji": mean_iters,
#             "średnia optymalnej wartości": mean_optimal_values,
#             "odchylenie iteracji ": std_iters,
#             "odchylenie optymalnej wartości": std_optimal_values,
#         },
#         ignore_index=True,
#     )

# alpha_10 = pd.DataFrame(
#     columns=[
#         "beta",
#         "średnia iteracji",
#         "średnia optymalnej wartości",
#         "odchylenie iteracji ",
#         "odchylenie optymalnej wartości",
#     ]
# )
# for beta in [0.05, 0.08, 0.1]:
#     mean_iters, mean_optimal_values, std_iters, std_optimal_values = test_alpha_beta(
#         alpha=10, beta=beta, k=25
#     )
#     alpha_10 = alpha_10._append(
#         {
#             "beta": beta,
#             "średnia iteracji": mean_iters,
#             "średnia optymalnej wartości": mean_optimal_values,
#             "odchylenie iteracji ": std_iters,
#             "odchylenie optymalnej wartości": std_optimal_values,
#         },
#         ignore_index=True,
#     )

# alpha_100 = pd.DataFrame(
#     columns=[
#         "beta",
#         "średnia iteracji",
#         "średnia optymalnej wartości",
#         "odchylenie iteracji ",
#         "odchylenie optymalnej wartości",
#     ]
# )
# for beta in [0.005, 0.008, 0.01]:
#     mean_iters, mean_optimal_values, std_iters, std_optimal_values = test_alpha_beta(
#         alpha=100, beta=beta, k=25
#     )
#     alpha_100 = alpha_100._append(
#         {
#             "beta": beta,
#             "średnia iteracji": mean_iters,
#             "średnia optymalnej wartości": mean_optimal_values,
#             "odchylenie iteracji ": std_iters,
#             "odchylenie optymalnej wartości": std_optimal_values,
#         },
#         ignore_index=True,
#     )

# alpha_1.to_csv("alpha_1.csv")
# alpha_10.to_csv("alpha_10.csv")
# alpha_100.to_csv("alpha_100.csv")
f3_lambda = lambda x: f3(np.array([x]))[0]
f19_lambda = lambda x: f19(np.array([x]))[0]
from autograd import grad
df = grad(f19_lambda)
# test_on_function(f=f19_lambda, beta=[0.00000000001])
# test_on_function(f=f19, beta=[0.1, 0.3, 0.5])


X_one = np.random.uniform(-10, 10, size=10)
print(f19_lambda(X_one))
print(df(X_one))
