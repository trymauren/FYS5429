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



def test():
    
    relu = Relu()
    sigmoid = Sigmoid()
    tanh = Tanh()

    z = np.ones((5,2))
    assert(relu(z).shape==z.shape)
    assert(tanh(z).shape==z.shape)
    assert(sigmoid(z).shape==z.shape)

    z = np.array([-100,-2,-1,0,1,2,100])
    assert(relu(z).all() == np.array([0,0,0,0,1,2,100]).all())
    assert(tanh(z).all() == np.array([-1.0,-0.9640275800758169,-0.7615941559557649,
                                0.0,0.7615941559557649,0.9640275800758169,1.0]).all())

    np.testing.assert_allclose(sigmoid(z) == np.array([0.0,0.11920292202211755,
                                                             0.26894142136999512,
                                                             0.5,
                                                             0.73105857863000487,
                                                             0.880797077977882444,
                                                             0.999999999999999]), 1e-10,1e10)

test()

