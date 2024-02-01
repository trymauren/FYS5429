import sys
import git
import numpy as np
from utils.activations import Relu, Tanh
from utils.loss_functions import Mean_Square_Loss as mse
from utils.optimisers import Adam
import utils.read_load_model
from collections.abc import Callable

path_to_root = git.Repo('.', search_parent_directories=True).working_dir
sys.path.append(path_to_root)


class ReccurentNN:

    def __init__(self,
                 hidden_activation: Callable = None,
                 output_activation: Callable = None,
                 loss_function: Callable = None,
                 optimiser: Callable = None
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
        self._loss = loss_function

        if not optimiser:
            optimiser = Adam()
        self._optimiser = optimiser

        """
        Initialize weights and biases as None until properly 
        initialized in fit() method
        """
        self.w_x, self.w_rec, self.w_y = None,None,None
        self.b_x, self.b_rec, self.b_y = None,None,None

    def _forward(self, x:np.array) -> np.array:
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
        # print(x.shape)
        n_time_steps = x.shape[0]
        # step_len = x.shape[1]
        step_len = 1
        # step len is thought to be the lenght of each time step
        # print('Time_steps:', n_time_steps)
        # h_layer_size = self.w_rec.shape[0]
        h_states = np.zeros((n_time_steps, step_len))
        # print('Shape of h_states:', h_states.shape)
        # h_states = np.zeros((n_time_steps, (n_time_steps, batch_size)))
        # print('Shape of x[0]:', x[0].shape)
        # print('Shape of w_x:', self.w_x.shape)
        z = x[0] @ self.w_x
        # print('Shape of z:', z.shape)
        h_t = self._hidden_activation(z)
        # print('Shape of h_t:', h_t.shape)
        h_states[0] = h_t
        for t in range(1, n_time_steps):
            inp = x[t] @ self.w_x
            # print('Shape of inp:', inp.shape)
            # print('Shape of h_states_-1:', h_states[t-1].shape)
            # print('Shape of b_rec:', self.b_rec.shape)
            # print('Shape of w_rec:', self.w_rec.shape)
            h_weighted = h_states[t-1] @ self.w_rec
            # print('Shape of h_weighted:', h_weighted.shape)
            z = inp + h_weighted
            # print('Shape of z:', z.shape)
            h_t = self._hidden_activation(z)
            # print('Shape of h_t:', h_t.shape)
            # print('Shape of h_states:', h_states.shape)
            h_states[t] = h_t

        y_predicted = h_states @ self.w_y
        print(y_predicted)
        return self._output_activation(y_predicted)

    def _backward(self, y_estimate: np.ndarray) -> None:
        pass

    def fit(self,
            X: np.ndarray,
            y: np.ndarray,
            epochs: int,
            improvement_threshold: float|int,
            # early_stopping_params,
            optimization_algorithm: str = None
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

        input_size = X.shape[-1]
        hidden_size = X.shape[-1]

        output_size = y.shape[1]

        # weight input -> hidden.
        self.w_x = np.random.randn(input_size, input_size)*0.01
        # weight hidden -> hidden
        self.w_rec = np.random.randn(input_size, input_size)*0.01
        # weight hidden -> output
        self.w_y = np.random.randn(output_size, input_size)*0.01

        # bias input -> hidden.
        self.b_x = np.random.randn(1)
        # bias hidden -> hidden.
        self.b_rec = np.random.randn(input_size, 1)
        # bias hidden -> output.
        self.b_y = np.random.randn(output_size, hidden_size)

        # Do a forward pass, return outputs from each hidden state 
        # and all current weights and biases
        y_predicted = self._forward(X[0])
        # Find current loss from predicted output values
        loss = self._loss(y, y_predicted)

        for e in range(epochs):
            # Do a backprogation and adjust current weights and biases to 
            # improve loss, return new improved weights and biases
        # self.w_xh, self.w_hh, self.w_hy, self.b_x, self.b_rec, \
        # self.b_y \ = self._backward(loss)                                             #Maybe return weights and biases in two tuples of three instead??? Might be a lot cleaner                    
            # Do a forward pass with new weights, return outputs from 
            # each hidden state and all the weights and biases
            y_predicted = self._forward(X[e])
            # Predict the current loss with used weights and biases
            loss = self._loss(y, y_predicted)

            # Calculate improvement, the rate or amount of the network
            # improves per epoch
            improvement = None                                                 #INSERT gradient thingy or improvement measure here
            # Check if improvement is significant enough according to the
            # set threshold, if not break and do an early exit
            if improvement < improvement_threshold:
                break
        
        read_load_model.save_model(self, "saved_models/", "RNN_model")         #Maybe have model/filename as a parameter? 


    def predict(self, X : np.ndarray) -> np.ndarray:
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
