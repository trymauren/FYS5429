import numpy as np
from scipy.special import expit # used for sigmoid
from abc import abstractmethod
# https://dustinstansbury.github.io/theclevermachine/derivation-common-neural-network-activation-functions

"""
Creds to "https://github.com/ddbourgin/numpy-ml/blob/master/numpy_ml/
neural_nets/activations/activations.py#L73-L137" for the setup
"""

class Activation():
    def __init__(self):
        super().__init__()

    def __call__(self, z):
        return self.eval(z)

    @abstractmethod
    def eval(self, z):
        raise NotImplementedError

    @abstractmethod
    def grad(self, z):
        raise NotImplementedError

class Relu(Activation):

    def __init__(self):
        super().__init__()

    def eval(self, z):
        """Returns a np.ndarray with same dimensions as z"""
        # ret = [n * (n > 0) for n in z]
        # return np.array(ret, dtype=float)
        return np.maximum(0,z)

    def grad(self, z):
        """Decided on using f'(0)=1. Could do f'(0)=0 instead"""
        return 1 if z >= 0 else 0

class Tanh(Activation):

    def __init__(self):
        super().__init__()

    def eval(self, z):
        """Returns a np.ndarray with same dimensions as z"""
        return np.tanh(z)

    def grad(self, z):
        """
        Assume z is the output from tanh(z)! or else the
        derivative must be calculated differently
        """
        return 1 - z^2

class Sigmoid(Activation): 

    def __init__(self):
        super().__init__()

    def eval(self, z):
        """Returns a np.ndarray with same dimensions as z"""
        return expit(z)

    def grad(self, z):
        """
        Assume z is the output from sigmoid(z)! or else the
        derivative must be calculated differently
        """
        return z*(1 - z)


