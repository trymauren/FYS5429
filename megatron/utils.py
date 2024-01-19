import numpy as np

def ReLU(x):
    return x * (x > 0)

def tanh():
    pass

def sigmoid():
    pass


def init_weights(input_size, hidden_layer_sizes, random_weight):
    layer_sizes = [input_size] + hidden_layer_sizes
    weight_matrices = [np.random.randn(layer_sizes[i], layer_sizes[i-1]) * random_weight for i in range(1,len(layer_sizes))]
    
    out = X@weight_matrices[0].T

np.random.seed(128)
init_weights(10,[2,3], 1)