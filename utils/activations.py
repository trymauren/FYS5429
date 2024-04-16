import numpy as np
import jax.numpy as jnp
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


class Identity(Activation):

    def __init__(self):
        super().__init__()

    def eval(self, z):
        return z


class Relu(Activation):

    def __init__(self):
        super().__init__()

    def eval(self, z):
        """Returns a jnp.ndarray with same dimensions as z"""
        # ret = [n * (n > 0) for n in z]
        # return jnp.array(ret, dtype=float)
        self.z = jnp.maximum(0, z)
        return self.z

    def grad(self, a):
        """Decided on using f'(0)=1. Could do f'(0)=0 instead"""
        return jnp.array([1 if i >= 0 else 0 for i in a])


class Tanh(Activation):

    def __init__(self):
        super().__init__()

    def eval(self, z):
        """Returns a jnp.ndarray with same dimensions as z"""
        return jnp.tanh(z)

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
        """Returns a jnp.ndarray with same dimensions as z"""
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
        softmax = jnp.exp(z)/jnp.sum(jnp.exp(z))
        return softmax

    def grad(self, a):
        """
        Assume a is the output from softmax(z)! or else the
        derivative must be calculated differently
        """
        s = a.reshape(-1, 1)
        return jnp.diagflat(s) - jnp.dot(s, s.T)

# soft = Softmax()
# data = [1/jnp.e,1/jnp.e,1/jnp.e]
# eval = soft.eval(data)
# print(eval)
# grad = soft.grad(eval)
# print(grad)

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


class Identity(Activation):

    def __init__(self):
        super().__init__()

    def eval(self, z):
        return z


class Relu(Activation):

    def __init__(self):
        super().__init__()

    def eval(self, z):
        """Returns a jnp.ndarray with same dimensions as z"""
        # ret = [n * (n > 0) for n in z]
        # return jnp.array(ret, dtype=float)
        self.z = jnp.maximum(0, z)
        return self.z

    def grad(self, a):
        """Decided on using f'(0)=1. Could do f'(0)=0 instead"""
        return jnp.array([1 if i >= 0 else 0 for i in a])


class Tanh(Activation):

    def __init__(self):
        super().__init__()

    def eval(self, z):
        """Returns a jnp.ndarray with same dimensions as z"""
        return jnp.tanh(z)

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
        """Returns a jnp.ndarray with same dimensions as z"""
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
        softmax = jnp.exp(z)/jnp.sum(jnp.exp(z))
        return softmax

    def grad(self, a):
        """
        Assume a is the output from softmax(z)! or else the
        derivative must be calculated differently
        """
        s = a.reshape(-1, 1)
        return jnp.diagflat(s) - jnp.dot(s, s.T)

# soft = Softmax()
# data = [1/jnp.e,1/jnp.e,1/jnp.e]
# eval = soft.eval(data)
# print(eval)
# grad = soft.grad(eval)
# print(grad)
