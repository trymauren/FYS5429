import sys
import git
import numpy as np
from collections.abc import Callable
import jax
from jax import grad
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
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
        y_true = jnp.array(y_true, dtype=jnp.float64)
        y_pred = jnp.array(y_pred, dtype=jnp.float64)
        if not nograd:
            self.y_pred = y_pred.copy()
            self.y_true = y_true.copy()
        loss = self.jax_loss(y_true, y_pred)
        return np.asarray(loss, dtype=y_pred.dtype)

    def grad(self):

        grad = (
                # 1
                2
                * np.subtract(self.y_pred, self.y_true)
                / self.y_pred.size
                # / len(self.y_pred)
                )
        # grad = grad.mean(axis=1)  # comment in to use reduction
        return grad

    def jax_loss(self, y_true, y_pred):
        return jnp.square(jnp.subtract(y_true, y_pred)).mean(dtype=y_pred.dtype)

    def grad_2(self):

        grad = jax.grad(self.jax_loss, argnums=0)
        return grad(self.y_true, self.y_pred)


class Classification_Logloss(LossFunction):

    def __init__(self):
        super().__init__()
        self.y_pred = None
        self.y_true = None
        self.probabilities = None

    def eval(self, y_true, y_pred, nograd):
        probabilities = y_pred.copy() + LOG_CONST
        if not nograd:
            self.y_pred = np.copy(y_pred)
            self.y_true = np.copy(y_true)
            self.probabilities = np.copy(probabilities)
        # y_true = y_true.astype(int)
        # loss = 0
        # for prob, true in zip(y_pred, y_true):
        #     for batch in range(y_pred.shape[1]):
        #         labels_correct = np.argwhere(true[batch])
        #         pred_correct = prob[batch][labels_correct]
        #         # print(pred_correct, np.log(pred_correct))
        #         # print()
        #         loss += np.log(pred_correct)
        # return -np.mean(loss)
        return -np.mean(np.sum(np.log(probabilities) * y_true))
        # return -np.mean(y_true*np.log(probabilities), dtype=y_pred.dtype)

    def grad(self):
        # See deep learning book, 10.18 for
        # explanation of the following line.
        grad = self.probabilities - self.y_true
        return grad

### MÅ FJERNE LOG CONST!!