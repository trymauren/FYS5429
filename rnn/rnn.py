import sys
import git
import numpy as np
from collections.abc import Callable
path_to_root = git.Repo('.', search_parent_directories=True).working_dir
sys.path.append(path_to_root)
from utils.activations import Relu, Tanh
from utils.loss_functions import Mean_Square_Loss as mse
from utils.optimisers import Adam
from utils import read_load_model


class ReccurentNN:

    def __init__(self,
                 hidden_activation: Callable = None,
                 output_activation: Callable = None,
                 loss_function: Callable = None,
                 optimiser: Callable = None,
                 name: str = 'rnn',
                 ) -> None:
        """Setting activation functions, loss function and optimiser"""
        if not hidden_activation:
            hidden_activation = Relu()
        self._hidden_activation = hidden_activation

        if not output_activation:
            output_activation = Tanh()
        self._output_activation = output_activation

        if not loss_function:
            loss_function = mse()
        self._loss_function = loss_function

        if not optimiser:
            optimiser = Adam()
        self._optimiser = optimiser

        """
        Initialize weights and biases as None until properly
        initialized in fit() method.
        xh = input  -> hidden
        hh = hidden -> hidden
        hy = hidden -> output
        """
        self.w_xh, self.w_hh, self.w_hy = None, None, None
        self.b_xh, self.b_hh, self.b_hy = None, None, None
        self.xs = None
        self.hs = None
        self.ys = None
        self.name = name

    def _forward(self, x: np.array):
        """
        Forward-pass method to be used in fit-method for training the
        RNN. Returns predicted output values

        Parameters:
        -------------------------------
        X : np.array
        - sequence of numbers to be used for prediction

        """

        n_time_steps = x.shape[0]
        step_len = x.shape[1]
        self.hs = np.zeros((n_time_steps, step_len))
        self.xs = np.zeros(x.shape)
        z = x[0] @ self.w_xh
        self.xs[0] = z
        h_t = self._hidden_activation(z)
        self.hs[0] = h_t
        # self.h_outputs = self.hs  # dummy variable
        for t in range(1, n_time_steps):
            x_weighted = x[t] @ self.w_xh
            h_weighted = self.hs[t-1] @ self.w_hh
            z = x_weighted + h_weighted
            self.xs[t] = z
            h_t = self._hidden_activation(z)
            self.hs[t] = h_t

        self.ys = self._output_activation(self.hs @ self.w_hy)
        return self.ys

    def _backward(self, y_true, y_pred: np.ndarray) -> None:

        # Set current loss. Not appliccable anymore?
        self._loss_function(y_true, y_pred)

        deltas_w_xh = np.zeros_like(self.w_xh, dtype=float)  # np.zeros?
        deltas_w_hh = np.zeros_like(self.w_hh, dtype=float)  # np.zeros?
        deltas_w_hy = np.zeros_like(self.w_hy, dtype=float)  # np.zeros?

        # deltas_b_xh = np.zeros_like(self.b_xh, dtype=float)
        deltas_b_hh = np.zeros_like(self.b_hh, dtype=float)
        deltas_b_hy = np.zeros_like(self.b_hy)
        prev_grad_h_Cost = np.zeros_like(self.hs[0].shape)
        # y_pred[t] = softmax(y_pred[t])

        # BACKPROPAGATION THROUGH TIME (BPTT):
        for t in range(len(self.hs)-1, -1, -1):

            """ BELOW IS CALCULATION OF GRADIENTS W/RESPECT TO HIDDEN_STATES
            - SEE (1-~20) IN TEX-DOCUMENT """

            """Just doing some copying. grad_o_Cost will, in the next
            line of code, contain the cost vector"""
            grad_o_Cost = np.copy(y_pred[t])

            """See deep learning book, 10.18 for
            explanation of following line. Also:
            http://cs231n.github.io/neural-networks-case-study/#grad
            Eventually, one can find grad(C) w/ respect to C^t"""
            grad_o_Cost[y_true[t]] -= 1

            """A h_state's gradient update are both influenced by the
            preceding h_state at time t+1, as well as the output at
            time t. The cost/loss of the current output derivated with
            respect to hidden state t is what makes up the following
            line before the "+ sign". After "+" is the gradient through
            previous hidden states and their outputs. This term after
            the "+" sign, is 0 for first step of BPTT.

            Eq. 16 in tex-document(see also eq. 15 for first iteration of BPPT)
            Eq. 10.20 in DLB"""
            grad_h_Cost = prev_grad_h_Cost + grad_o_Cost @ self.w_hy

            """The following line is to shorten equations. It fetches/
            differentiates the hidden activation function."""
            d_act = self._hidden_activation.grad(self.hs[t])

            """ BELOW IS CALCULATION OF GRADIENT W/RESPECT TO WEIGHTS """

            """Cumulate the error."""
            deltas_w_hy += grad_o_Cost @ self.hs[t]         # 10.24 in DLB
            deltas_w_hh += grad_h_Cost @ d_act * self.hs[t-1]  # 10.26 in DLB
            deltas_w_xh += grad_h_Cost @ d_act * self.xs[t]    # 10.28 in DLB

            """Pass on the bits of the chain rule to the calculation of
            the previous hidden state update

            This line equals the first part of eq. 10.21 in DLB
            To emphasize: before the "+" sign in 10.21 in DLB"""
            prev_grad_h_Cost = d_act @ self.w_hh @ grad_h_Cost

            # Biases:
            deltas_b_hy += grad_o_Cost * 1     # 10.22 in DLB
            deltas_b_hh += grad_h_Cost @ d_act    # 10.22 in DLB
            # deltas_b_xh += 0                   # no bias on input right?

        # Weight updates:
        self.w_hy += deltas_w_hy
        self.w_hh += deltas_w_hh
        self.w_xh += deltas_w_xh
        # Bias updates
        self.w_hy += deltas_b_hy
        self.w_hh += deltas_b_hh
        # self.w_xh += deltas_b_xh

    def fit(self,
            X: np.ndarray,
            y: np.ndarray,
            epochs: int,
            improvement_threshold: float | int,
            # early_stopping_params,
            ) -> None:
        """
        Method for training the RNN, iteratively runs _forward(),
        _loss(), and _backwards() to predict values, find loss and
        adjust weights until a given number of training epochs has been
        reached or the degree of improvement between epochs is below a
        set threshold.

        Parameters:
        -------------------------------
        X : np.array
        - input sequence, sequence elements may be scalars
          or vectors.

        y : np.array
        - true output values to compare predicted results
          against to calculate loss.

        epochs: int
        - number of epochs to train for.

        improvement_threshold : float | int
        - threshold for minimum improvement per epoch before early exit

        optimization_algorithm : str
        - Method for optimization to run during training????????????????

        Returns:
        -------------------------------
        None
        """

        # Fetch the inner dimension (-1), which corresponds to the length
        # of each time step in a sequence, if that makes sense?
        time_step_len = X.shape[-1]
        hidden_size = time_step_len
        output_size = y.shape[1]  # this may not be correct

        # TODO: move these to a function and add random configurability
        self.w_xh = np.random.randn(time_step_len, time_step_len)*0.01
        self.w_hh = np.random.randn(time_step_len, time_step_len)*0.01
        self.w_hy = np.random.randn(time_step_len, time_step_len)*0.01
        # TODO: move these to a function and add random configurability
        self.b_xh = np.random.randn(1)
        self.b_hh = np.random.randn(time_step_len, 1)
        self.b_hy = np.random.randn(output_size, hidden_size)

        # Do a forward pass
        y_pred = self._forward(X[0])

        for e in range(epochs):
            if e >= X.shape[0]:  # temporary
                break
            # Do a backprogation and adjust current weights and biases to
            # improve loss, return new improved weights and biases
            self._backward(y[e], y_pred)
            # Do a forward pass with new weights, return outputs from
            # each hidden state and all the weights and biases
            y_pred = self._forward(X[e])

            # TODO: add stopping criterias
        read_load_model.save_model(self, 'saved_models/', self.name)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the next value in a sequence of given inputs to the RNN
        network

        Parameters:
        -------------------------------
        X : np.array
        - sequence of values to have the next one predicted

        Returns:
        -------------------------------
        np.array
        - Predicted next value for the given input sequence
        """
        return self._forward(X)[-1]
