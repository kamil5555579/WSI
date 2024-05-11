import numpy as np
from layer import Layer, lelu, linear, sigmoid, tanh
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from solver_evo import EvolutionaryAlgorithm
import abc

class MLP:
    def __init__(self, layers):
        self.layers = layers
    
    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X
    
    @abc.abstractmethod
    def train(self, X, y):
        pass

class gdMLP(MLP):
    def __init__(self, layers, learning_rate=0.0001):
        super().__init__(layers)
        self.learning_rate = learning_rate

    def backward(self, d_output):
        for layer in reversed(self.layers):
            d_output = layer.backward(d_output, self.learning_rate)

    def train(self, X, y, epochs=1000):
        for epoch in range(epochs):
            # Przesyłanie sygnału do przodu
            output = self.forward(X)
            
            # Obliczanie błędów
            error = y - output # pochodna funkcji straty
            d_output = error

            # Przesyłanie sygnału wstecz
            self.backward(d_output)

            if epoch % 100 == 0:
                mse = np.mean((y - output) ** 2)
                print(f"Epoka {epoch} - Błąd MSE: {mse}")

class evolutionaryMLP(MLP):
    def __init__(self, layers, population_size=10, generations=100, sigma=0.1):
        super().__init__(layers)
        self.population_size = population_size
        self.generations = generations
        self.sigma = sigma

    def train(self, X, y):
        parameters = self.layers_to_parameters()
        P0 = np.random.uniform(-1, 1, size=(self.population_size, parameters.size))
        f = lambda x: self.func_to_optimize(x, X, y)
        solver = EvolutionaryAlgorithm(f, P0, self.population_size, self.sigma)
        x_t, f_t = solver.solve()
        best = x_t[-1]
        print(f_t)
        print(best)
        self.layers = self.parameters_to_layers(best)

    def func_to_optimize(self, parameters_population, X, y):
        loss = []
        for parameters in parameters_population:
            self.layers = self.parameters_to_layers(parameters)
            output = self.forward(X)
            loss.append(np.mean((y - output) ** 2))
        return np.array(loss)
        
    def layers_to_parameters(self):
        weights = [layer.weights for layer in self.layers]
        biases = [layer.bias for layer in self.layers]
        return np.concatenate([w.ravel() for w in weights] + [b for b in biases])
    
    def parameters_to_layers(self, parameters):
        weights = []
        biases = []
        start = 0
        for layer in self.layers:
            end = start + layer.weights.size
            weights.append(parameters[start:end].reshape(layer.weights.shape))
            start = end
            end = start + layer.bias.size
            biases.append(parameters[start:end])
            start = end
        for i, layer in enumerate(self.layers):
            layer.weights = weights[i]
            layer.bias = biases[i]
        return self.layers
    

if __name__ == "__main__":
    
    layers = [
        Layer(1, 20, activation=tanh),
        # Layer(20, 20, activation=lelu),
        Layer(20, 1, activation=linear),
    ]

    # mlp = evolutionaryMLP(layers, population_size=10, generations=100, sigma=0.1)
    mlp = gdMLP(layers, learning_rate=0.005)

    # Generowanie danych dla funkcji do aproksymacji (np. sinus)
    X = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)  # Dane wejściowe
    y = np.sin(X)  # Dane wyjściowe

    # Standaryzacja danych wejściowych
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Trenowanie modelu
    # mlp.train(X_scaled, y)
    mlp.train(X_scaled, y, epochs=5000)
    print(mlp.layers[0].weights), print(mlp.layers[0].bias, mlp.layers[1].weights, mlp.layers[1].bias)

    # Testowanie modelu
    output = mlp.forward(X_scaled)

    # Wizualizacja wyników
    plt.plot(X, y, label="Dane oryginalne")
    plt.plot(X, output, label="Dane z modelu")
    plt.legend()
    plt.show()
