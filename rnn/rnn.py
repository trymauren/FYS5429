import numpy as np

class RecurrentNN:

    def __init__(self) -> None:
        pass

    def _forward(
            self, X:np.array, 
            W_x:np.array, W_rec:np.array, W_y: np.array, 
            b_x:np.array, b_rec:np.array, b_y: np.array
        ) -> np.array:
        """
        Forward-pass method to be used in fit-method for training the RNN. 
        Returns a predicted output value which is used to calculate loss
        which is later used to adjust backpropagation for weight correction during training. 

        Parameters:
        ------------------------------
            X : np.array - input sequence, sequence of several time
                dependent datapoints, often in the for of vectors (np.arrays) 
            
            W_x : np.array - input weights, from input layer to hidden layer

            W_rec : np.array - recurrent weights, from hidden layer back onto itself

            W_y : np.array - output weights, from hidden layer to output layer
        
        Returns:
        ------------------------------
            y_predicted : np.array - predicted output values from each hidden state
        """
        time_steps = X.shape[0]
        h_layer_size = W_rec.shape[0]

        h_states = np.zeros(time_steps,h_layer_size)        
        
        z = X[t,:]@W_x + b_x
        h_t = self.act(z)
        h_states[0,:] = np.flatten(h_t)

        for t in range(1,time_steps):
            z = X[t,:]@W_x + h_states[t-1,:]@W_rec + b_rec
            h_t = self.act(z)
        
        h_states[t,:] = np.flatten(h_t)

        y_predicted = h_states[-1,:]@W_y + b_y

        return y_predicted
        

    def _backward(self, y_estimate : np.ndarray) -> None:
        pass

    def fit(self,
            X : np.ndarray,
            y : np.ndarray,
            epochs : int,
            improvement_threshold : float|int,
            # early_stopping_params,
            optimization_algorithm : str = None
        ) -> None:

        """
        Method for training the RNN, iteratively runs _forward(), _loss(),
        and _backwards() to predict values, find loss and adjust weights until
        a given number of training epochs has been reached or the degree of 
        improvement between epochs is below a set threshold.

        Parameters:
        ------------------------------
        """
        
        input_size = X.shape[1]
        hidden_size = X.shape[0]
        output_size = y.shape[1]


        W_xh = np.random.randn(hidden_size, input_size)*0.01     # weight input -> hidden. 
        W_hh = np.random.randn(hidden_size, hidden_size)*0.01   # weight hidden -> hidden
        W_hy = np.random.randn(output_size, hidden_size)*0.01     # weight hidden -> output

        y_predicted = self._forward(X,W_xh,W_hh)
        loss = self._loss(y, y_predicted)

        for e in epochs:
            self._backward(loss)
            y_predicted = self._forward(X,W_xh,W_hh)
            loss = self._loss(y, y_predicted)

            improvement = None #INSERT gradient thingy here
            if improvement < improvement_threshold:
                break

    def predict(self, X : np.ndarray) -> np.ndarray:
        pass