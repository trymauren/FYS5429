import sys
import git
import numpy as np
path_to_root = git.Repo('.', search_parent_directories=True).working_dir
sys.path.append(path_to_root)


class Optimiser():

    def __init__(learning_rate: float = 0.003):
        self.learning_rate = learning_rate


class SGD(Optimiser):

    def __init__(
            self,
            learning_rate: float = 0.003,
            ):

        super().__init__()
        self.learning_rate = learning_rate
        self.update = np.zeros(num_params)

    def step(self, params):
        self.update = self.learning_rate * params


class SGD_momentum(Optimiser):

    def __init__(
            self,
            learning_rate: float = 0.003,
            momentum_coef: float = 0.1,
            ):

        super().__init__()
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.update = 0

    def step(self, params):
        momentum = self.momentum_rate*self.update
        self.update = momentum + self.learning_rate*params
        return self.update


class AdaGrad(Optimiser):

    def __init__(
            self,
            epsilon=1e-15,
            ):

        super().__init__()
        self.momentum = momentum
        self.epsilon = epsilon
        self.sum_squared_gradients = np.zeros(num_params)

    def step(self, params):
        self.sum_squared_gradients += np.square(params)
        self.alpha = np.sqrt(sum_squared_gradients)
        # correct to use @?
        self.update = self.learning_rate / (alpha+self.epsilon) * params
        return self.update
