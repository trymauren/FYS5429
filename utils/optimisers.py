import sys
import git
import numpy as np
from abc import abstractmethod
path_to_root = git.Repo('.', search_parent_directories=True).working_dir
sys.path.append(path_to_root)


class Optimiser():

    def __init__(self):
        pass

    def __call__(self, params, **kwargs):
        return self.step(params, **kwargs)


class SGD(Optimiser):

    def __init__(self):
        super().__init__()

    def step(self, params, learning_rate=0.001):
        self.learning_rate = learning_rate
        self.update = [0]*len(params)
        for idx, param in enumerate(params):
            self.update[idx] = self.learning_rate*param
        return self.update


class SGD_momentum(Optimiser):

    def __init__(self):
        super().__init__()
        self.update = None

    def step(self, params, learning_rate=0.001, momentum_rate=0.9):
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        if self.update is None:
            self.update = [0]*len(params)

        momentum = [0]*len(params)
        for idx, param in enumerate(params):
            momentum = self.momentum_rate*self.update[idx]
            self.update[idx] = momentum+self.learning_rate*param
        return self.update


class AdaGrad(Optimiser):

    def __init__(self):
        super().__init__()
        self.lambda_ = 1e-7
        self.alphas = None
        self.update = None

    def step(self, params, learning_rate=0.001):
        self.learning_rate = learning_rate
        if self.alphas is None:
            self.alphas = [0]*len(params)
            self.update = [0]*len(params)

        for idx, param in enumerate(params):
            self.alphas[idx] += np.square(param)
            adagrad = param / (np.sqrt(self.lambda_ + self.alphas[idx]))
            self.update[idx] = self.learning_rate * adagrad
        return self.update


class RMSProp(Optimiser):

    def __init__(self):
        super().__init__()
        self.lambda_ = 1e-6
        self.update = None
        self.alphas = None

    def step(self, params, learning_rate=0.001, decay_rate=0.001):
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        if self.alphas is None:
            self.alphas = [0]*len(params)
            self.update = [0]*len(params)

        for idx, param in enumerate(params):
            self.alphas[idx] += (
                                 self.decay_rate * param
                                 + (1 - decay_rate)
                                 * np.square(param)
                                )
            rmsprop = param / np.sqrt(self.lambda_ + self.alphas[idx])
            self.update[idx] = self.learning_rate * rmsprop
        return self.update


def clip_gradient(gradients: np.ndarray, threshold: float) -> np.ndarray:
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
    # for gradient in gradients:
    #     gradient

    for g in gradients:
        norm_g = np.linalg.norm(g, ord=1)
        if norm_g > threshold:
            g *= threshold/norm_g
    return gradients