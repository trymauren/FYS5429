from importlib.resources import open_text
from pathlib import Path
import sys
import os
from typing import Dict
import git
import numpy as np
from collections.abc import Callable
import yaml
from utils.activations import Relu, Tanh
from utils.loss_functions import Mean_Square_Loss as mse
from utils.optimisers import SGD, SGD_momentum, AdaGrad
from utils import read_load_model
from utils.word_embedding import word_embedding
import matplotlib.pyplot as plt
from tqdm import tqdm

path_to_root = git.Repo('.', search_parent_directories=True).working_dir
sys.path.append(path_to_root)
np.random.seed(24)


class RNN:

    def __init__(
            self,
            hidden_activation: Callable = None,
            output_activation: Callable = None,
            loss_function: Callable = None,
            optimiser: Callable = None,
            regression: bool = False,
            classification: bool = False,
            name: str = 'rnn',
            config: Dict | Path | str = 'default',
            seed: int = 24,
            ) -> None:

        np.random.seed(seed)
        # Setting activation functions, loss function and optimiser
        if not hidden_activation:
            hidden_activation = Relu()
        self._hidden_activation = eval(hidden_activation)
        if not output_activation:
            output_activation = Tanh()
        self._output_activation = eval(output_activation)

        if not loss_function:
            loss_function = mse()
        self._loss_function = eval(loss_function)

        if not optimiser:
            optimiser = AdaGrad()
        self._optimiser = eval(optimiser)

        self.regression = regression
        self.classification = classification
        # Initialize weights and biases as None until properly
        # initialized in fit() method.
        # xh = input  -> hidden
        # hh = hidden -> hidden
        # hy = hidden -> output
        self.w_xh, self.w_hh, self.w_hy = None, None, None

        self.b_hh, self.b_hy = None, None

        self.xs, self.hs, self.ys = None, None, None

        self.built = False

        self.name = name

        self.stats = {
            'other stuff': [],
        }

    def windowed_data(
            self,
            X: np.ndarray,
            num_hidden_states: int
            ) -> np.ndarray:

        split_X = []
        split_y = []
        for t in range(len(X)-num_hidden_states-1):
            split_X.append(X[t:(t + num_hidden_states)])
            split_y.append(X[t+1: t+num_hidden_states+1])

        return split_X, split_y

    def format_data(
            self,
            X,
            is_text: bool
            ) -> None:

        formatted_X = None
        if is_text:
            formatted_X = word_embedding.read_txt(X)
        else:
            formatted_X = []
            for data in zip(X):
                formatted_X.append(float(data))
            formatted_X = np.array(formatted_X)
        return formatted_X

    def _forward(
            self,
            X_partition,
            num_hidden_states) -> None:
        """
        Forward-pass method to be used in fit-method for training the
        RNN. Returns predicted output values

        Parameters:
        -------------------------------
        X_partition:
            - A partition of samples

        num_hidden_states : int
            - Number of times to unroll the rnn architecture

        Returns:
        -------------------------------
        None
        """
        for t in range(num_hidden_states):
            x_weighted = self.w_xh @ X_partition[t]
            h_weighted = self.w_hh @ self.hs[t-1]
            z = x_weighted + h_weighted
            self.xs[t] = z
            h_t = self._hidden_activation(z)
            self.hs[t] = h_t
            self.ys[t] = self._output_activation(self.w_hy @ self.hs[t])
        return self.ys

    def _backward(self, num_backsteps=10) -> None:

        deltas_w_xh = np.zeros_like(self.w_xh, dtype=float)
        deltas_w_hh = np.zeros_like(self.w_hh, dtype=float)
        deltas_w_hy = np.zeros_like(self.w_hy, dtype=float)

        deltas_b_hh = np.zeros_like(self.b_hh, dtype=float)
        deltas_b_hy = np.zeros_like(self.b_hy, dtype=float)

        prev_grad_h_Cost = np.zeros_like(self.num_hidden_nodes)

        loss_grad = self._loss_function.grad()
        num_backsteps = min(len(self.hs)-1, num_backsteps)
        for t in range(num_backsteps, -1, -1):

            """ BELOW IS CALCULATION OF GRADIENTS W/RESPECT TO HIDDEN_STATES
            - SEE (1-~20) IN TEX-DOCUMENT """

            """OUTDATED"""
            # """Just doing some copying. grad_o_Cost will, in the next
            # line of code, contain the cost vector"""
            # grad_o_Cost = np.copy(y_pred[t])

            # """See deep learning book, 10.18 for
            # explanation of following line. Also:
            # http://cs231n.github.io/neural-networks-case-study/#grad
            # Eventually, one can find grad(C) w/ respect to C^t"""
            # grad_o_Cost[y_true[t]] -= 1
            """OUTDATED END"""

            """ NEW """
            # grad_o_Cost = self._loss_function.grad()
            if self.regression:
                grad_o_Cost_t = loss_grad[:, t]
            if self.classification:
                print('not implemented error')
            """ NEW END """

            """A h_state's gradient update are both influenced by the
            preceding h_state at time t+1, as well as the output at
            time t. The cost/loss of the current output derivated with
            respect to hidden state t is what makes up the following
            line before the "+ sign". After "+" is the gradient through
            previous hidden states and their outputs. This term after
            the "+" sign, is 0 for first step of BPTT.

            Eq. 16 in tex-document(see also eq. 15 for first iteration of BPPT)
            Eq. 10.20 in DLB"""
            grad_h_Cost = prev_grad_h_Cost + self.w_hy.T @ grad_o_Cost_t
            # print(prev_grad_h_Cost.shape)
            # print(self.w_hy.T.shape)
            # print(grad_h_Cost.shape)
            """The following line is to shorten equations. It fetches/
            differentiates the hidden activation function."""
            d_act = self._hidden_activation.grad(self.hs[t])

            """ BELOW IS CALCULATION OF GRADIENT W/RESPECT TO WEIGHTS """

            """Cumulate the error."""
            deltas_w_hy += self.hs[t].T * grad_o_Cost_t  # 10.24 in DLB
            deltas_w_hh += d_act @ self.hs[t-1] * grad_h_Cost  # 10.26 in DLB
            deltas_w_xh += d_act @ self.xs[t] * grad_h_Cost  # 10.28 in DLB
            deltas_b_hy += grad_o_Cost_t * 1  # 10.22 in DLB
            deltas_b_hh += d_act @ grad_h_Cost  # 10.22 in DLB

            """Pass on the bits of the chain rule to the calculation of
            the previous hidden state update
            This line equals the first part of eq. 10.21 in DLB
            To emphasize: the part before the "+" in 10.21 in DLB"""
            prev_grad_h_Cost = d_act @ self.w_hh.T @ grad_h_Cost

        # # Weight updates:
        # self.w_hy += 0.01 * deltas_w_hy
        # self.w_hh += 0.01 * deltas_w_hh
        # self.w_xh += 0.01 * deltas_w_xh
        # # Bias updates
        # self.w_hy += 0.01 * deltas_b_hy
        # self.w_hh += 0.01 * deltas_b_hh
        params = [self.w_hy, self.w_hh, self.w_xh,
                  self.b_hy, self.b_hh]
        deltas = [deltas_w_hy, deltas_w_hh, deltas_w_xh,
                  deltas_b_hy, deltas_b_hh]
        steps = self._optimiser(deltas)

        for param, step in zip(params, steps):
            param -= step

    def fit(self,
            X: np.ndarray = None,
            y: np.ndarray = None,
            epochs: int = None,
            batches: int = 5,
            is_text: bool = False,
            learning_rate: float = 0.01,
            num_hidden_states: int = 5,
            num_hidden_nodes: int = 5,
            num_backsteps: float = 0.01,
            ) -> None:
        """
        Method for training the RNN, iteratively runs _forward(), and
        _backwards() to predict values, find loss and adjust weights
        until a given number of training epochs has been reached.

        Parameters:
        -------------------------------
        X : np.array, shape: m x n
            - Input sequence, sequence elements may be scalars
              or vectors.
            - m: number of samples
            - n: number of features (for text, this corresponds to number
                                    ´of entries in embedding vector)
_
        y : np.array, shape: m x 1
            - Labels
            - m: equal to n in X    (for text, this corresponds to number
                                     of entries in embedding vector)

        epochs: int
            - Number of iterations to train for
                                    (1 epoch = iterate through all samples
                                     in X)

        num_hidden_states : int
            - Number of times to unroll the rnn architecture

        num_hidden_nodes : int
            - Number of fully connected layers to add

        Returns:
        -------------------------------
        None
        """

        X = np.array(X)
        y = np.array(y)

        examples, time_steps, num_features = X.shape
        examples, time_steps_y, output_size = y.shape

        self.output_size = output_size
        self.num_features = num_features
        self.num_hidden_nodes = num_hidden_nodes

        self.init_states(time_steps)

        self.init_weights()

        X_split = np.split(X, batches)
        y_split = np.split(y, batches)

        self.stats['loss'] = [0]*epochs

        for e in tqdm(range(epochs)):

            for x_batch, y_batch in zip(X_split, y_split):

                y_pred_batch = np.zeros_like(y_batch)

                for idx, (x, y) in enumerate(zip(x_batch, y_batch)):
                    y_pred = self._forward(
                        x,
                        num_hidden_states
                    )
                    y_pred_batch[idx] = y_pred

                    self.loss(self.ys, y, e)

                self._backward(
                    num_backsteps=10
                )

        read_load_model.save_model(  # pickle dump the trained estimator
            self,
            'saved_models/',
            self.name
        )

        # return self._forward(
        #     X,
        #     y,
        #     num_hidden_states
        # )

    def predict(self, X: np.ndarray) -> np.ndarray:
        # TODO: add some assertions/ checks
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
        X = np.array(X)
        examples, time_steps, features = X.shape
        assert (self.num_features == features)  # cannot change number of features at predict time
        self.init_states(time_steps)
        ret = np.zeros((examples, self.output_size))
        for example in range(examples):
            prev_h = np.copy(self.hs[-1])
            y_pred = self._forward(
                X[example],
                time_steps
            )
            self.hs[-1] = prev_h
            ret[example] = y_pred[-1]

        return ret

    def init_weights(
            self,
            scale=0.1) -> None:
        """
        Initialises weights and biases and assign them to instance variables.

        Parameters:
        -------------------------------
        scale : float
            - scaling of init weights
        Returns:
        -------------------------------
        None
        """
        # Notes:
        # w_xh = 1 x n, x = n x 1, => z = 1 x 1 (per state)
        # w_hh = hidden_layers x hidden_layers = 1 x 1, h = 1 x 1 (per state)
        # w_hy = 1 x hidden_layers = 1 x 1, y = 1 x 1 (per state)
        self.w_xh = np.random.randn(
            self.num_hidden_nodes, self.num_features) * scale
        self.w_hh = np.random.randn(
            self.num_hidden_nodes, self.num_hidden_nodes) * scale
        self.w_hy = np.random.randn(
            self.output_size, self.num_hidden_nodes) * scale

        self.b_hh = np.random.randn(
            self.num_hidden_nodes, self.num_hidden_nodes) * scale
        self.b_hy = np.random.randn(
            self.output_size, self.num_hidden_nodes) * scale

    def init_states(
            self,
            time_steps,
            ) -> None:
        """
        Initialises states and assign them to instance variables.

        Parameters:
        -------------------------------
        num_hidden_states : int
            - Number of times to unroll the rnn architecture.
              This essentially means how many forward and backward steps
              are performed per data-partition (per batch).

        num_hidden_nodes : int
            - Number of fully connected layers to add

        Returns:
        -------------------------------
        None
        """
        if self.built:
            prev_h = self.hs[-1]
        self.hs = np.zeros((time_steps, self.num_hidden_nodes))
        if self.built:
            self.hs[-1] = prev_h
        self.xs = np.zeros((time_steps, self.num_hidden_nodes))
        self.ys = np.zeros((time_steps, self.output_size))
        self.built = True

    def loss(self, y_true, y_pred, epoch):
        loss = self._loss_function(y_true, y_pred)
        self.stats['loss'][epoch] += np.mean(loss)

    def get_stats(self):
        return self.stats
