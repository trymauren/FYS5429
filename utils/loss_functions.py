import sys
import git
path_to_root = git.Repo('.', search_parent_directories=True).working_dir
sys.path.append(path_to_root)

import numpy as np
from collections.abc import Callable 


LOG_CONST = 1e-15 # why this number? Many use it

class LossFunction(Callable):

    def __call__(self, y, y_pred):
        return self.eval(y, y_pred)

class Mean_Square_Loss(LossFunction):
    
    def __init__(self):
        super().__init__()

    def eval(self, y, y_pred):
        # this may be the sklearn implementation?
        return np.square(np.subtract(y, y_pred)).mean()

class Classification_Logloss(LossFunction):
    
    def __init__(self, n_classes):
        self.n_classes = n_classes
        super().__init__()

    def eval(self, y, y_pred):
        y_pred = y_pred + LOG_CONST # to avoid log(0) calculations
        return -np.sum(y*np.log(y_pred))



# # remove this later
# y_true = np.array([2,4,5,1,2,4])
# y_pred = np.array([1,3,5,2,2,3])
# unique = np.unique(y_true)
# mcl = Classification_logloss(unique)
# loss = mcl(y_true,y_pred)
# print('Classification logloss: {0}'.format(loss))

# msl = Mean_Square_loss()
# mse = msl(y_true,y_pred)
# print('Regression logloss: {0}'.format(mse))
