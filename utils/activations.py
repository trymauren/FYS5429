import numpy as np
from scipy.special import expit  # used for sigmoid
from collections.abc import Callable
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


class Relu(Activation):

    def __init__(self):
        super().__init__()

    def eval(self, z):
        return self.eval(z)

    def eval(self, z):
        """Returns a np.ndarray with same dimensions as z"""
        # ret = [n * (n > 0) for n in z]
        # return np.array(ret, dtype=float)
        self.z = np.maximum(0, z)
        return self.z

    def grad(self, a):
        """Decided on using f'(0)=1. Could do f'(0)=0 instead"""
        return np.array([1 if i >= 0 else 0 for i in a])


class Tanh(Activation):

    def __init__(self):
        super().__init__()

    def eval(self, z):
        return self.eval(z)

    def eval(self, z):
        """Returns a np.ndarray with same dimensions as z"""
        return np.tanh(z)

    def grad(self, a):
        """
        Assume a is the output from tanh(a)! or else the
        derivative must be calculated differently
        """
        return 1 - a**2


class Sigmoid(Activation):

    def __init__(self):
        super().__init__()

    def eval(self, z):
        return self.eval(z)

    def eval(self, z):
        """Returns a np.ndarray with same dimensions as z"""
        return expit(z)

    def grad(self, a):
        """
        Assume a is the output from sigmoid(z)! or else the
        derivative must be calculated differently
        """
        return a*(1 - a)


class Softmax(Activation):

    def __init__(self):
        super().__init__()

    def eval(self, z):
        softmax = np.exp(z)/np.sum(np.exp(z))
        return softmax

    def grad(self, a):
        """
        Assume a is the output from softmax(z)! or else the
        derivative must be calculated differently
        """
        s = a.reshape(-1, 1)
        return np.diagflat(s) - np.dot(s, s.T)

# soft = Softmax()
# data = [1/np.e,1/np.e,1/np.e]
# eval = soft.eval(data)
# print(eval)
# grad = soft.grad(eval)
# print(grad)
