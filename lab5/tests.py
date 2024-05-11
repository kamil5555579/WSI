import numpy as np

input_size = 5
output_size = 3

weights = np.random.randn(input_size, output_size)
bias = np.random.randn(output_size)

X = np.random.randn(10, input_size)

output = np.dot(X, weights) + bias

def relu(x, derivative=False):
    if derivative:
        return np.where(x > 0, 1, 0)
    return np.maximum(0, x)
activation = np.maximum(0, output)
activation_2 = relu(output)

print(X)
print(weights)
print(bias)
print(output)
print(activation)
print(activation_2)
print(weights.ravel())