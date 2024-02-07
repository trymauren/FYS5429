import sys
import git
import numpy as np
from utils.activations import Relu, Tanh
from utils.loss_functions import Mean_Square_Loss as mse
from utils.optimisers import Adam
from utils import read_load_model
from collections.abc import Callable

path_to_root = git.Repo('.', search_parent_directories=True).working_dir
sys.path.append(path_to_root)


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

    def _forward(self, x: np.array) -> np.array:
        """
        Forward-pass method to be used in fit-method for training the
        RNN. Returns a predicted output value which is used to calculate
        loss which is later used to adjust backpropagation for weight
        correction during training.

        Parameters:
        -------------------------------
        X : np.array
        - sequence of numbers to be used for prediction

        w_x : np.array
        - input weights, from input layer to hidden layer

        w_rec : np.array
        - recurrent weights, from hidden layer back onto itself

        w_y : np.array
        - output weights, from hidden layer to output layer

        Returns:
        -------------------------------
        y_predicted : np.array
        - predicted output values from each hidden state
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

        self.outputs = self._output_activation(self.hs @ self.w_hy)

        return self.outputs

    def _backward(self, y_true, y_pred: np.ndarray) -> None:

        self._loss_function(y_true, y_pred)  # set current loss

        deltas_w_xh = np.zeros_like(self.w_xh, dtype=float)  # np.zeros?
        deltas_w_hh = np.zeros_like(self.w_hh, dtype=float)  # np.zeros?
        deltas_w_hy = np.zeros_like(self.w_hy, dtype=float)  # np.zeros?

        deltas_b_xh = np.zeros_like(self.b_xh, dtype=float)
        deltas_b_hh = np.zeros_like(self.b_hh, dtype=float)
        deltas_b_hy = np.zeros_like(self.b_hy)
        prev_dh = np.zeros_like(self.hs[0].shape)
        # y_pred[t] = softmax(y_pred[t])

        for t in range(len(self.hs)-1, -1, -1):

            """Fetch the loss of this time-step"""
            d_loss = np.copy(y_pred[t])
            print(d_loss)
            print(d_loss[0])
            """See deep learning book, 10.18 for
            explanation of following line. Also:
            http://cs231n.github.io/neural-networks-case-study/#grad"""
            d_loss[y_true[t]] -= 1

            """Adjustments to output weights is simple; the derivative of
            the cost function with respect to the output weights"""

            deltas_w_hy += d_loss @ self.hs[t] #Adjustment amount of output weights (accumulates over time to get final adjustment after all time has passed)

            """A h_state's gradient update are both influenced by the
            next h_state at time t+1, as well as the output at time t.
            The cost/loss of the current output derivated with respect to hidden
            state t is what makes up the following line before the "+ sign".
            hidden state. After "+" is the influence from previous hidden
            states and their outputs."""
            dh = d_loss @ self.w_hy + prev_dh   #Weight adjustment needed for current hidden state with regards to this states output and previous (next in forward pass) states influence on loss

            """ The following line is to shorten equations. It fetches the
            gradient of hidden state t."""
            d_act = self._hidden_activation.grad(self.hs[t]) #Gradient of current hidden state, to be used to adjust recurrent and input weights alongside dh which is adjustment amount 
                                                             #influenced by current hidden states output and previous hidden states (????????????????)

            """Cumulate the error."""
            deltas_w_hh += dh @ d_act * self.hs[t-1] #Adjustment amount of hidden (recurrent) weights (accumulates over time to get final amount after all time has passed)
            deltas_w_xh += dh @ d_act * self.xs[t] #Adjustment amount of input weights (accumulates over time to get final amount after all time has passed)

            """Pass on the bits of the chain rule to the calculation of
            the previous hidden state update"""
            prev_dh = d_act @ self.w_hh @ dh

            # Biases:
            deltas_b_hy += d_loss * 1
            deltas_b_hh += dh @ d_act
            deltas_b_xh += 0  # change this

        ret = deltas_w_hh, deltas_w_hy, deltas_b_xh, deltas_b_hh, deltas_b_hy
        return ret  # or update weights?

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

        # Do a forward pass, return outputs from each hidden state
        y_pred = self._forward(X[0])

        # Find current loss from predicted output values
        # loss = self._loss(y, y_predicted)
        # exit()
        for e in range(epochs):
            if e >= X.shape[0]:  # temporary
                break
            # Do a backprogation and adjust current weights and biases to 
            # improve loss, return new improved weights and biases
            self._backward(y[e], y_pred)
            # Do a forward pass with new weights, return outputs from 
            # each hidden state and all the weights and biases
            y_pred = self._forward(X[e])
            # Predict the current loss with used weights and biases
            print(y_pred)
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

    def update_weights(self):
        pass
