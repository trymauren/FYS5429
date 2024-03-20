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
from utils.optimisers import Adam
from utils import read_load_model
from utils.word_embedding import word_embedding
import matplotlib.pyplot as plt

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
            ) -> None:

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
            optimiser = Adam()
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

        self.name = name

    def windowed_data(self, X : np.ndarray, num_hidden_states : int) ->\
    np.ndarray:
        split_X = []
        split_y = []
        for t in range(len(X)-num_hidden_states-1):
            split_X.append(X[t:(t + num_hidden_states)])
            split_y.append(X[t+1: t+num_hidden_states+1])

        return split_X, split_y
    
    def format_data(self, X, is_text : bool):
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
            y_partition,
            num_hidden_states) -> None:
        """
        Forward-pass method to be used in fit-method for training the
        RNN. Returns predicted output values

        Parameters:
        -------------------------------
        X_partition:
            - A partition of samples

        y_partition:
            - A partition of labels

        num_hidden_states : int
            - Number of times to unroll the rnn architecture

        Returns:
        -------------------------------
        None
        """
        None
        for t in range(num_hidden_states):
            x_weighted = self.w_xh @ X_partition[t]
            h_weighted = self.w_hh @ self.hs[t-1]
            z = x_weighted + h_weighted
            self.xs[t] = z
            h_t = self._hidden_activation(z)
            self.hs[t] = h_t
            self.ys[t] = self._output_activation(self.w_hy @ self.hs[t])
            #Added check if y is None to avoid calculating loss when predicting
            #and also have a consistent value to pass a y parameter when 
            #calling forward() in predict
            if y_partition is not None:
                self._loss_function(self.ys[t], y_partition[t])
        return self.ys

    
    def _backward(
            self,
            y_true,
            y_pred,
            num_hidden_nodes) -> None:

        deltas_w_xh = np.zeros_like(self.w_xh, dtype=float)
        deltas_w_hh = np.zeros_like(self.w_hh, dtype=float)
        deltas_w_hy = np.zeros_like(self.w_hy, dtype=float)

        deltas_b_hh = np.zeros_like(self.b_hh, dtype=float)
        deltas_b_hy = np.zeros_like(self.b_hy, dtype=float)

        prev_grad_h_Cost = np.zeros_like(num_hidden_nodes)

        loss = self._loss_function(y_true, y_pred)
        loss_grad = self._loss_function.grad(loss=loss)

        for t in range(len(self.hs)-1, -1, -1):

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

        # Weight updates:
        self.w_hy += deltas_w_hy
        self.w_hh += deltas_w_hh
        self.w_xh += deltas_w_xh
        # Bias updates
        self.w_hy += deltas_b_hy
        self.w_hh += deltas_b_hh
        # self._optimiser(
        #         deltas_w_hy=deltas_w_hy,
        #         deltas_w_hh=deltas_w_hh,
        #         deltas_w_xh=deltas_w_xh,
        #         deltas_b_hy=deltas_b_hy,
        #         deltas_b_hh=deltas_b_hh,
        #     )

    def fit(self,
            X: np.ndarray = None,
            y: np.ndarray = None,
            epochs: int = None,
            num_hidden_states: int = 5,
            num_hidden_nodes: int = 5,
            is_text : bool = False,
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

        time_steps, num_features = X.shape
        time_steps_y, output_size = y.shape

        assert (time_steps == time_steps_y)
        assert (num_hidden_states <= time_steps)

        self.init_states(
            output_size,
            num_hidden_states,
            num_hidden_nodes=num_hidden_nodes
        )
        self.init_weights(
            num_features,
            output_size,
            num_hidden_nodes=num_hidden_nodes
        )

        # Split data into "batches"
        #TODO exchange with windowed_data() ??????? Update:
        #It fucked up, has to look through again
        partitions = np.floor(time_steps/num_hidden_states)
        X_split = np.split(X, partitions, axis=0)
        y_split = np.split(y, partitions, axis=0)

        #X_split,y_split = self.windowed_data(X,num_hidden_states)


        # Run training
        for e in range(epochs):
            print(self.w_xh)
            for X_partition, y_partition in zip(X_split, y_split):
                y_pred = self._forward(
                    X_partition,
                    y_partition,
                    num_hidden_states
                )
                print(y_pred)
                self._backward(
                    y_partition,
                    y_pred,
                    num_hidden_nodes
                )

        read_load_model.save_model(  # pickle dump the trained estimator
            self,
            'saved_models/',
            self.name
        )

        return self._forward(
            X,
            y,
            num_hidden_states
        )

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

        #Have to pass Y (as None to avoid calculating loss when only 
        #doing a prediction (less computational power required *sunglass
        #emoji*)) and number of hidden nodes to make forward() run
        return self._forward(X,None, self.num_hidden_nodes)[-1]

    def init_weights(
            self,
            num_features,
            output_size,
            num_hidden_nodes=1,
            scale=0.1) -> None:
        """
        Initialises weights and biases and assign them to instance variables.

        Parameters:
        -------------------------------
        num_features : int
            - Number of entries in each time-sample (time-step)

        output_size : int
            - Number of entries in each label

        num_hidden_nodes : int
            - Number of fully connected layers to add

        Returns:
        -------------------------------
        None
        """
        # Notes:
        # w_xh = 1 x n, x = n x 1, => z = 1 x 1 (per state)
        # w_hh = hidden_layers x hidden_layers = 1 x 1, h = 1 x 1 (per state)
        # w_hy = 1 x hidden_layers = 1 x 1, y = 1 x 1 (per state)
        self.w_xh = np.random.randn(
            num_hidden_nodes, num_features) * scale
        self.w_hh = np.random.randn(
            num_hidden_nodes, num_hidden_nodes) * scale
        self.w_hy = np.random.randn(
            output_size, num_hidden_nodes) * scale

        self.b_hh = np.random.randn(
            num_hidden_nodes, num_hidden_nodes) * scale
        self.b_hy = np.random.randn(
            output_size, num_hidden_nodes) * scale

    def init_states(
            self,
            output_size,
            num_hidden_states,
            num_hidden_nodes=1) -> None:
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
        self.num_hidden_nodes = num_hidden_nodes
        self.hs = np.zeros((num_hidden_states, num_hidden_nodes))
        self.xs = np.zeros((num_hidden_states, num_hidden_nodes))
        self.ys = np.zeros((num_hidden_states, output_size))
        