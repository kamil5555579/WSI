import numpy as np
import matplotlib.pyplot as plt
from network import gdMLP, evolutionaryMLP
from layer import Layer, lelu, linear, sigmoid, tanh
from sklearn.preprocessing import StandardScaler

def f(x):
    return x**2 * np.sin(x) + 100 * np.sin(x) * np.cos(x)

X = np.linspace(-10, 10, 100).reshape(-1, 1)
y = f(X)

# Standaryzacja danych wejściowych

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

scaler_y = StandardScaler()
y = scaler_y.fit_transform(y)

# Definicja modelu
layers = [
    Layer(1, 100, activation=lelu),
    # Layer(20, 20, activation=lelu),
    Layer(100, 1, activation=linear),
]

# mlp = evolutionaryMLP(layers, population_size=50, generations=100, sigma=0.5)
mlp = gdMLP(layers, learning_rate=0.005)

# Trenowanie modelu
# mlp.train(X_scaled, y)
mlp.train(X_scaled, y, epochs=10000)

# Testowanie modelu
output = mlp.forward(X_scaled)

# Wizualizacja wyników
plt.plot(X, y, 'bo', label='Dane')
plt.plot(X, output, 'ro', label='Aproksymacja')

plt.legend()

plt.show()