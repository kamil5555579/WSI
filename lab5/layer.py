import numpy as np

def lelu(x, derivative=False):
    if derivative:
        return np.where(x > 0, 1, 0.1)
    return np.where(x > 0, x, 0.1 * x)

def linear(x, derivative=False):
    if derivative:
        return np.ones_like(x)
    return x

def sigmoid(x, derivative=False):
    s = 1 / (1 + np.exp(-x))
    if derivative:
        return s * (1 - s)
    return s

def tanh(x, derivative=False):
    t = np.tanh(x)
    if derivative:
        return 1 - t ** 2
    return t

class Layer:
    def __init__(self, input_size, output_size, activation=lelu):
        self.weights = np.random.uniform(-1/input_size, 1/input_size, (input_size, output_size))
        self.bias = np.zeros(output_size)
        self.activation = activation

    def forward(self, X):
        self.input = X
        self.output = np.dot(X, self.weights) + self.bias
        self.output = self.activation(self.output)
        return self.output
    
    def backward(self, d_output, learning_rate):
        # # Obliczanie gradientu wejścia
        # d_input = np.dot(d_output, self.weights.T) * self.activation(self.input, derivative=True)
        
        # # Aktualizacja wag i biasów
        # self.weights += np.dot(self.input.T, d_output) * learning_rate
        # self.bias += np.sum(d_output, axis=0) * learning_rate
        
        d_output = d_output * self.activation(self.output, derivative=True)
        d_input = np.dot(d_output, self.weights.T)
        self.weights += np.dot(self.input.T, d_output) * learning_rate
        self.bias += np.sum(d_output, axis=0) * learning_rate
        
        return d_input
    
    def __repr__(self):
        return f"Layer: {self.weights.shape[0]}x{self.weights.shape[1]}"

    
# Test
if __name__ == "__main__":
    X = np.random.randn(5, 1)
    activation = lelu
    layer = Layer(1, 3, activation)
    output = layer.forward(X)
    d_output = np.random.randn(5, 3)
    d_input = layer.backward(d_output, 0.01)
    print(X)
    print(output)
    print(d_output)
    print(d_input)
    print(layer.weights)
    print(layer.bias)
