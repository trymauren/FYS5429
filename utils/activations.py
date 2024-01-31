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
        self.z = np.maximum(0,z)
        return self.z

    def grad(self, z):
        """Decided on using f'(0)=1. Could do f'(0)=0 instead"""
        return 1 if z >= 0 else 0

class Tanh(Activation):
    
    self.z = None

    def __init__(self):
        super().__init__()

    def eval(self, z):
        """Returns a np.ndarray with same dimensions as z"""
        self.z = np.tanh(z)
        return self.z 

    def grad(self, z):
        """
        Assume z is the output from tanh(z)! or else the
        derivative must be calculated differently
        """
        if self.z == None:
            print('Error, something should be thrown here')
            return None
            
        self.z = None
        return 1 - self.z^2

class Sigmoid(Activation): 

    def __init__(self):
        super().__init__()

    def eval(self, z):
        """Returns a np.ndarray with same dimensions as z"""
        return expit(z)

    def grad(self, a):
        """
        Assume z is the output from sigmoid(z)! or else the
        derivative must be calculated differently
        """
        return a*(1 - a)


