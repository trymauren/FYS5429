import sys
import git
import numpy as np
from collections.abc import Callable

path_to_root = git.Repo('.', search_parent_directories=True).working_dir
sys.path.append(path_to_root)

LOG_CONST = 1e-15  # why this number? Many use it


class LossFunction(Callable):

    def __call__(self, y, y_pred, nograd=False):
        return self.eval(y, y_pred, nograd=nograd)


class Mean_Square_Loss(LossFunction):

    def __init__(self):
        super().__init__()
        self.y_pred = None
        self.y_true = None

    def eval(self, y_true, y_pred, nograd=False):
        if not nograd:
            self.y_pred = y_pred
            self.y_true = y_true

        loss = np.square(np.subtract(y_true, y_pred)).mean()
        return loss

    def grad(self):
        grad = (2
                * np.subtract(self.y_pred, self.y_true)
                # * np.subtract(self.y_pred, self.y_true)
                / len(self.y_pred))

        return grad


class Classification_Logloss(LossFunction):

    def __init__(self):
        super().__init__()
        self.y_pred = None
        self.y_true = None
        self.probabilities = None

    def eval(self, y_true, y_pred, nograd):
        y_pred += LOG_CONST
        probabilities = y_pred
        if not nograd:
            self.y_pred = y_pred
            self.y_true = y_true
            self.probabilities = probabilities

        return -np.mean(y_true*np.log(probabilities))

    def grad(self):
        probabilities = np.copy(self.probabilities)
        # See deep learning book, 10.18 for
        # explanation of the following line.
        grad = probabilities - self.y_true
        return grad
