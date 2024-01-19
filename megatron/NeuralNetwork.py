from typing import Callable, List
import utils

class NeuralNetwork:

    def __init__(self, hidden_layer_sizes: List[int], activation_function: str, alpha: int, solver: str):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation_function = activation_function
        self.alpha = alpha
        self.solver = solver

        # Initialize the weights and biases
        self.weights = None
        self.biases = None
        self.init_layers()
        self.init_biases()

        # Placeholder for inputs and outputs
        self.X = None
        self.out = None

    def init_layers(self):
        # Initialize weights for each layer
        layer_sizes = [self.X.shape[1]] + self.hidden_layer_sizes + [self.out.shape[1]]
        self.weights = [np.random.randn(layer_sizes[i], layer_sizes[i + 1]) for i in range(len(layer_sizes) - 1)]

    def init_biases(self):
        # Initialize biases for each layer
        self.biases = [np.random.randn(1, size) for size in self.hidden_layer_sizes + [self.out.shape[1]]]

    def forward(self, X):
        for W in self.weights:
            z = np.dot(activation, weight) + bias
            activation = self.activation_function(z)
        
        self.out = activation
        return self.out

    def backward(self, Y):
        # Backward pass through the network
        # This is a placeholder for the backpropagation algorithm
        pass

    def fit(self, X, Y, epochs):
        # Fit the neural network to the data
        self.X = X
        self.weights = init_weights(X.shape[0],hidden_layer_sizes)

        # Update weights and biases according to the solver
        pass