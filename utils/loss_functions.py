import sys
import git
import numpy as np
from collections.abc import Callable

path_to_root = git.Repo('.', search_parent_directories=True).working_dir
sys.path.append(path_to_root)

LOG_CONST = 1e-15  # why this number? Many use it


class LossFunction(Callable):

    def __call__(self, y, y_pred):
        return self.eval(y, y_pred)


class Mean_Square_Loss(LossFunction):

    def __init__(self):
        super().__init__()
        self.loss = None
        self.y_pred = None
        self.y_true = None

    def eval(self, y_true, y_pred):
        self.y_pred = y_pred
        self.y_true = y_true
        loss = np.square(np.subtract(y_true, y_pred)).mean(axis=0)
        if self.loss is None:
            self.loss = loss
        else:
            self.loss += loss
        return self.loss

    # not tested and verified:
    def grad(self):
        grad = (2
                * np.array([np.subtract(self.y_pred, self.y_true)]).T
                / len(self.y_pred))
        self.loss = None
        return grad


class Classification_Logloss(LossFunction):

    def __init__(self):
        super().__init__()

    def eval(self, y_true, y_pred):
        y_pred += LOG_CONST  # to avoid log(0) calculations
        self.y_pred = y_pred
        self.y_true = y_true
        self.probabilities = np.exp(y_pred)/np.sum(np.exp(y_pred))
        return -np.sum(y_true*np.log(self.probabilities))

    def grad(self):
        probabilities = np.copy(self.probabilities)
        # for t in range(len(probabilities)):
        #     probabilities[t] -= self.y_true[t]
        probabilities -= self.y_true
        return probabilities.T
