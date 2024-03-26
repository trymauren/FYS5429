import sys
import git
import numpy as np
from abc import abstractmethod
path_to_root = git.Repo('.', search_parent_directories=True).working_dir
sys.path.append(path_to_root)


class Optimiser():

    def __init__(
            self,
            ):
        pass

    def __call__(self, params, learning_rate=None):
        return self.step(params, learning_rate)


class SGD(Optimiser):

    def __init__(
            self,
            ):
        super().__init__()

    def step(self, params, learning_rate):
        self.learning_rate = learning_rate
        self.update = [0]*len(params)
        for idx, param in enumerate(params):
            self.update[idx] = self.learning_rate*param
        return self.update


class SGD_momentum(Optimiser):

    def __init__(
            self,
            momentum_rate: float = 0.001,
            ):

        super().__init__()
        self.momentum_rate = momentum_rate
        self.update = None

    def step(self, params, learning_rate):
        self.learning_rate = learning_rate
        if self.update is None:
            self.update = [0]*len(params)

        momentum = [0]*len(params)
        for idx, param in enumerate(params):
            momentum = self.momentum_rate*self.update[idx]
            self.update[idx] = momentum+self.learning_rate*param
        return self.update


class AdaGrad(Optimiser):

    def __init__(
            self,
            epsilon=1e-15,
            ):

        super().__init__()
        self.epsilon = epsilon
        self.alphas = None
        self.update = None

    def step(self, params, learning_rate):
        self.learning_rate = learning_rate
        if self.alphas is None:
            self.alphas = [0]*len(params)
            self.update = [0]*len(params)

        for idx, param in enumerate(params):
            self.alphas[idx] += np.square(param)
            coef = np.sqrt(self.alphas[idx] + self.epsilon)
            adagrad = param / coef
            self.update[idx] = self.learning_rate * adagrad

        return self.update


def clip_gradient(gradient_vector: np.ndarray, threshold: float) -> np.ndarray:
    """
    Finds l2-norm of gradient vector and normalizes it.
    TODO Find out if actual delta parameters are the ones to be adjusted 
    to make norm of grad vector be within threshold, or if just scaling 
    the grad vector itself suffices
    EDIT: found what seems to be an answer to exactly how the clipping 
    is done, it seems it's only scaling of the actual gradient: 
    https://towardsdatascience.com/what-is-gradient-clipping-b8e815cdfb48
    g = g*(threshold/l2norm(g)) or g = threshold*(g/l2norm(g))
    """
    grad_norm_col = np.linalg.norm(gradient_vector, ord=2, axis=1)
    grad_norm = np.linalg.norm(grad_norm_col, ord=2, axis=0)
    #grad_norm = np.sqrt(sum((np.sum(g**2)) for g in gradient_vector)) #NOT RIGHT
    
    #Only need positive threshold check as l2 norm ensues we only get 
    #positive norm values
    if grad_norm > threshold:
        gradient_vector = gradient_vector * float(threshold/grad_norm)
    else:
        return gradient_vector
    return gradient_vector