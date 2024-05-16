import numpy as np
import matplotlib.pyplot as plt
from network import gdMLP, evolutionaryMLP
from layer import Layer, lelu, linear, sigmoid, tanh
from sklearn.preprocessing import StandardScaler

def f(x):
    return x**2 * np.sin(x) + 100 * np.sin(x) * np.cos(x)

def test_gdMLP(n: int, X_scaled: np.ndarray, y: np.ndarray, epochs: int, learning_rate: float):
    # Definicja modelu
    layers = [
        Layer(1, n, activation=lelu),
        Layer(n, n, activation=lelu),
        Layer(n, 1, activation=linear),
    ]

    # mlp = evolutionaryMLP(layers, population_size=50, generations=1000, sigma=0.001)
    mlp = gdMLP(layers, learning_rate=0.002)

    # Trenowanie modelu
    # mlp.train(X_scaled, y)
    mlp.train(X_scaled, y, epochs=25000)

    # Testowanie modelu
    output = mlp.forward(X_scaled)

    mse = np.mean((y - output) ** 2)

    # Wizualizacja wyników
    # plt.plot(X_scaled, y, 'bo', label='Dane')
    # plt.plot(X_scaled, output, 'ro', label='Aproksymacja')

    # plt.legend()

    # plt.show()

    return mse

def test_evolutionaryMLP(n: int, X_scaled: np.ndarray, y: np.ndarray, generations: int, population_size: int, sigma: float):

    # Definicja modelu
    layers = [
        Layer(1, n, activation=lelu),
        Layer(n, n, activation=lelu),
        # Layer(n, n, activation=lelu),
        Layer(n, 1, activation=linear),
    ]

    mlp = evolutionaryMLP(layers, population_size=population_size, generations=generations, sigma=sigma)

    # Trenowanie modelu
    mlp.train(X_scaled, y)

    # Testowanie modelu
    output = mlp.forward(X_scaled)

    mse = np.mean((y - output) ** 2)

    # Wizualizacja wyników
    # plt.plot(X_scaled, y, 'bo', label='Dane')
    # plt.plot(X_scaled, output, 'ro', label='Aproksymacja')

    # plt.legend()

    # plt.show()

    return mse

if __name__ == "__main__":

    X = np.linspace(-10, 10, 100).reshape(-1, 1)
    y = f(X)

    # Standaryzacja danych wejściowych

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    scaler_y = StandardScaler()
    y = scaler_y.fit_transform(y)

    n = 100
    epochs = 25000
    learning_rate = 0.02

    # mses_gdMLP = np.zeros(10)
    # for i in range(10):
    #     mse_gdMLP = test_gdMLP(n, X_scaled, y, epochs, learning_rate)
    #     mses_gdMLP[i] = mse_gdMLP
    # print(f"gdMLP: {mses_gdMLP}")
    # print(f"gdMLP mean: {np.mean(mses_gdMLP)}")
    # print(f"gdMLP std: {np.std(mses_gdMLP)}")
    # print(f"gdMLP min: {np.min(mses_gdMLP)}")
    # print(f"gdMLP max: {np.max(mses_gdMLP)}")

    generations = 1000
    population_size = 50
    sigma = 0.002

    mses_evolutionaryMLP = np.zeros(10)
    for i in range(10):
        mse_evolutionaryMLP = test_evolutionaryMLP(n, X_scaled, y, generations, population_size, sigma)
        mses_evolutionaryMLP[i] = mse_evolutionaryMLP
    print(f"evolutionaryMLP: {mses_evolutionaryMLP}")
    print(f"evolutionaryMLP mean: {np.mean(mses_evolutionaryMLP)}")
    print(f"evolutionaryMLP std: {np.std(mses_evolutionaryMLP)}")
    print(f"evolutionaryMLP min: {np.min(mses_evolutionaryMLP)}")
    print(f"evolutionaryMLP max: {np.max(mses_evolutionaryMLP)}")