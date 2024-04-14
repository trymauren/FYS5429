from importlib.resources import open_text
from pathlib import Path
import sys
import os
from typing import Dict
import git
import numpy as np
from collections.abc import Callable
import yaml
from utils.activations import Relu, Tanh, Identity, Softmax
from utils.loss_functions import Mean_Square_Loss as mse
from utils.loss_functions import Classification_Logloss as ce
from utils import optimisers
from utils.optimisers import SGD, SGD_momentum, AdaGrad, RMSProp
from utils import read_load_model
import utils.text_processing as text_proc
from utils.text_processing import WORD_EMBEDDING
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm

path_to_root = git.Repo('.', search_parent_directories=True).working_dir
sys.path.append(path_to_root)


class RNN:

    def __init__(
            self,
            hidden_activation: Callable = None,
            output_activation: Callable = None,
            loss_function: Callable = None,
            optimiser: Callable = None,
            name: str = 'rnn',
            config: Dict | Path | str = 'default',
            seed: int = 24,
            clip_threshold: float = 5,
            **optimiser_params,
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
        self.optimiser_params = optimiser_params

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

        self.clip_threshold = clip_threshold

        self.stats = {
            'other stuff': [],
        }

    def _forward(
            self,
            x_sample,
            generate=False,
            nograd=False,
            output_probabilities=False,
            ) -> None:
        """
        Forward-pass method to be used in fit-method for training the
        RNN. Returns predicted output values

        Parameters:
        -------------------------------
        x_sample:
            - A sample of vectors

        generate:
            - Whether to insert output at time t=1 as input at time t=2

        Returns:
        -------------------------------
        None
        """
        def onehot_to_embedding(vec):
            return vec

        xs = np.zeros_like(self.xs)
        hs = np.zeros_like(self.hs)
        ys = np.zeros_like(self.ys)

        for t in range(self.num_hidden_states):
            x_weighted = self.w_xh @ x_sample[t]
            h_weighted = self.w_hh @ self.hs[t-1]
            a = self.b_hh + h_weighted + x_weighted
            xs[t] = a
            h = self._hidden_activation(a)
            hs[t] = h
            # why the fuck is squeeze required here?
            o = self.b_hy + self.w_hy @ h.squeeze()
            y = self._output_activation(o)
            ys[t] = y

            if not nograd:
                self.xs[t] = xs[t]
                self.hs[t] = hs[t]
                self.ys[t] = ys[t]

            if generate:
                if t < self.num_hidden_states - 1:
                    if output_probabilities:
                        ys[t] = onehot_to_embedding(np.argmax(ys[t]))
                    x_sample[t+1] = ys[t]

        return ys

    def _backward(self, num_backsteps=np.inf) -> None:

        deltas_w_xh = np.zeros_like(self.w_xh, dtype=float)
        deltas_w_hh = np.zeros_like(self.w_hh, dtype=float)
        deltas_w_hy = np.zeros_like(self.w_hy, dtype=float)

        deltas_b_hh = np.zeros_like(self.b_hh, dtype=float)
        deltas_b_hy = np.zeros_like(self.b_hy, dtype=float)

        prev_grad_h_Cost = np.zeros_like(self.num_hidden_nodes)

        #NOTE: Implemented gradient clipping, however shape error, 
        #      gradient norm is a list of floats, not one number
        loss_grad = self._loss_function.grad()

        num_backsteps = min(len(self.hs)-1, num_backsteps)
        for t in range(num_backsteps, -1, -1):

            """ BELOW IS CALCULATION OF GRADIENTS W/RESPECT TO HIDDEN_STATES """
            grad_o_Cost_t = loss_grad[:, t]
            # grad_h_Cost = optimisers.clip_gradient(grad_h_Cost, self.clip_threshold)
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
            deltas_b_hy += grad_o_Cost_t.T * 1  # 10.22 in DLB
            deltas_b_hh += d_act @ grad_h_Cost  # 10.22 in DLB

            """Pass on the bits of the chain rule to the calculation of
            the previous hidden state update
            This line equals the first part of eq. 10.21 in DLB
            To emphasize: the part before the "+" in 10.21 in DLB"""
            prev_grad_h_Cost = d_act @ self.w_hh.T @ grad_h_Cost

        params = [self.w_hy, self.w_hh, self.w_xh,
                  self.b_hy, self.b_hh]
        deltas = [deltas_w_hy, deltas_w_hh, deltas_w_xh,
                  deltas_b_hy, deltas_b_hh]
        clipped_deltas = optimisers.clip_gradient(deltas, self.clip_threshold)
        # steps = self._optimiser(deltas, **self.optimiser_params)

        steps = self._optimiser(clipped_deltas, **self.optimiser_params)

        for param, step in zip(params, steps):
            param -= step

    def fit(self,
            X: np.ndarray = None,
            y: np.ndarray = None,
            epochs: int = None,
            num_hidden_nodes: int = 5,
            num_backsteps: int = None,
            return_sequences: bool = False,
            independent_samples: bool = True,
            ) -> np.ndarray:
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

        learning_rate: float,

        num_hidden_nodes : int
            - Number of fully connected hidden nodes to add

        num_backsteps : int
            - Number of hidden states to backpropagate through

        return_sequences : bool
            - Whether to return content of all output states (self.ys)
              or only the last output states. Shape:
              If True:
              shape = (num_hidden_states, time_steps, output_size)
              If False:
              shape = (num_hidden_states, output_size)

        return_sequences : bool
            - Whether to reset initial hidden state between each processed
              sample in X.

        Returns:
        -------------------------------
        (np.ndarray, np.ndarray) = (output states, hidden state)

        """
        X = np.array(X, dtype=object)  # object to allow inhomogeneous shape
        y = np.array(y, dtype=object)  # object to allow inhomogeneous shape

        if X.ndim != 3:
            raise ValueError("Input data for X has to be of 3 dimensions:\
                             Samples x time steps x features")
        if y.ndim != 3:
            raise ValueError("Input data for y has to be of 3 dimensions:\
                             Samples x time steps x features")
        print("Please wait, training model:")

        _, _, num_features = X.shape
        _, _, output_size = y.shape

        self.output_size = output_size
        self.num_features = num_features
        self.num_hidden_nodes = num_hidden_nodes

        self._init_weights()

        self.stats['loss'] = np.zeros(epochs)

        for e in tqdm(range(epochs)):
            for sample_x, sample_y in zip(X, y):
                self.num_hidden_states = len(sample_x)
                if independent_samples:
                    self._init_states()
                else:
                    prev_h = None
                    if self.hs is not None:
                        prev_h = self.hs[-1]
                    self._init_states()
                    if prev_h is not None:
                        self.hs[-1] = prev_h

                y_pred = self._forward(
                    np.array(sample_x, dtype=float),
                    generate=False,
                    nograd=False,
                )

                self._loss(np.array(sample_y, dtype=float), y_pred, e)

                self._backward(num_backsteps=num_backsteps)

        read_load_model.save_model(  # pickle dump the trained estimator
            self,
            'saved_models/',
            self.name
        )
        print("Training complete, proceed")
        return self.ys, self.hs[-1]

    def predict(
            self,
            X: np.ndarray,
            time_steps_to_generate: int = 1,
            ) -> np.ndarray:
        """
        Predicts the next value in a sequence of given inputs to the RNN
        network

        Parameters:
        -------------------------------
        x_seed : np.array
        - An X-sample to seed generation of samples

        h_seed : np.array
        - Hidden state value to seed generation of samples

        Returns:
        -------------------------------
        np.ndarray
        - Generated next samples
        """
        # if h_seed is None:
        #     self.hs[-1] = np.zeros_like(self.hs[-1])
        # else:
        #     self.hs[-1] = h_seed
        if X.ndim != 3:
            raise ValueError("Input data for X has to be of 3 dimensions:\
                             Samples x time steps x features")

        self.num_hidden_states = time_steps_to_generate
        self._init_states()
        # X = np.zeros((time_steps_to_generate, len(x_seed)))
        # X[0] = x_seed
        _, _, vec_length = X.shape
        X_gen = np.zeros((time_steps_to_generate, vec_length))
        X_gen[0] = X[-1][-1]
        for x in X[:-1]:  # seeding the generation
            self._forward(np.array(x, dtype=float))
        ret = self._forward(np.array(X_gen, dtype=float), generate=True, output_probabilities=True)
        return ret

    def _init_weights(self) -> None:
        """
        Initialises weights and biases and assign them to instance variables.

        Parameters:
        -------------------------------
        None
        Returns:
        -------------------------------
        None
        """
        self.w_xh = np.random.uniform(
            -0.3, 0.3, size=(self.num_hidden_nodes, self.num_features))
        self.w_hh = np.random.uniform(
            -0.3, 0.3, size=(self.num_hidden_nodes, self.num_hidden_nodes))
        self.w_hy = np.random.uniform(
            -0.3, 0.3, size=(self.output_size, self.num_hidden_nodes))

        self.b_hh = np.random.uniform(
            -0.3, 0.3, size=(1, self.num_hidden_nodes))
        self.b_hy = np.random.uniform(
            -0.3, 0.3, size=(1, self.output_size))

    def _init_states(self) -> None:
        """
        Initialises states and assign them to instance variables.

        Parameters:
        -------------------------------
        None
        Returns:
        -------------------------------
        None
        """
        self.hs = np.zeros((self.num_hidden_states, self.num_hidden_nodes))
        self.xs = np.zeros((self.num_hidden_states, self.num_hidden_nodes))
        self.ys = np.zeros((self.num_hidden_states, self.output_size))

    def _loss(self, y_true, y_pred, epoch):
        loss = self._loss_function(y_true, y_pred)
        self.stats['loss'][epoch] += np.mean(loss)

    def plot_loss(self, plt, figax=None, savepath=None, show=False):
        # Some config stuff
        if figax is None:
            fig, ax = plt.subplots()
        else:
            fig, ax = figax
        ax.set_yscale('symlog')
        ax.set_yticks([5, 10, 20, 50, 100, 200, 500, 1000])
        ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.plot(
                self.stats['loss'],
                label=str(self._optimiser.__class__.__name__))

        ax.legend()
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.set_title('Training loss')
        if savepath:
            plt.savefig(savepath)
        if show:
            plt.show()
        return fig, ax

    def get_stats(self):
        return self.stats