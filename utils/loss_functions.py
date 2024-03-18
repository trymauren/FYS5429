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
        self.loss = np.square(np.subtract(y_true, y_pred)).mean(axis=0)
        return self.loss

    # not tested and verified:
    def grad(self, loss=None):
        if self.loss is None and loss is None:
            print('raise error')  # TODO
        elif not (loss is None):
            self.loss = loss
        grad = (2
                * np.array([np.subtract(self.y_pred, self.y_true)]).T
                / len(self.y_pred))
        return grad

# class Classification_Logloss(LossFunction):

#     def __init__(self, n_classes):
#         self.n_classes = n_classes
#         super().__init__()

#     def eval(self, y, y_pred):
#         y_pred = y_pred + LOG_CONST  # to avoid log(0) calculations
#         return -np.sum(y*np.log(y_pred))
