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
        self.y = None
        self.y_pred = None
        self.loss = None

    def eval(self, y, y_pred):
        self.y = y
        self.y_pred = y_pred
        self.loss = np.square(np.subtract(y, y_pred)).mean()
        return self.loss

    def grad(self):
        if self.loss is None:
            print('raise error')  # TODO
        else:
            # The thought is that we are using batches for training
            return 2*self.loss / self.y_pred.shape[1]  # this may be wrong

class Classification_Logloss(LossFunction):

    def __init__(self, n_classes):
        self.n_classes = n_classes
        super().__init__()

    def eval(self, y, y_pred):
        y_pred = y_pred + LOG_CONST  # to avoid log(0) calculations
        return -np.sum(y*np.log(y_pred))
