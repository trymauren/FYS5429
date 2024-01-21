import numpy as np

def ReLU(x):
    ret = [n * (n > 0) for n in x]
    return np.array(ret, dtype=float)


def tanh():
    pass

def sigmoid():
    pass


def init_weights(layer_sizes, random_weight) -> list:
    """
    Initialise weights.

    The size of the input (and the preceding hidden layer) is what determines
    the number the weights for the first layer and so on.
    """
    weight_matrices = [np.random.randn(layer_sizes[i], layer_sizes[i-1]) * random_weight for i in range(1,len(layer_sizes))]
    return weight_matrices
    
def init_biases(layer_sizes, random_weight) -> list:
    """
    Initialise biases.

    One bias is initialised per node per layer.
    """
    bias_matrices = [np.random.randn(1, layer_sizes[i]) * random_weight for i in range(1, len(layer_sizes))]
    return bias_matrices


# np.random.seed(128)
# init_weights(10,[2,3], 1)

