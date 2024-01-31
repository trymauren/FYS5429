import numpy as np
from utils import activations, loss_functions, read_load_model

class RecurrentNN:

    def __init__(self,
                 activation: str, 
                 loss: str
        ) -> None:

        self._act = activations.activation
        self._loss = loss_functions.loss

        #Initialize weights as nothing until properly initialized in 
        #fit() method
        self.w_x, self.w_rec, self.w_y = None
        #Initialize biases as nothing until properly initialized in 
        #fit() method
        self.b_x, self.b_rec, self.b_y = None


    def _forward(self, X:np.array) -> np.array:
        """
        Forward-pass method to be used in fit-method for training the 
        RNN. Returns a predicted output value which is used to calculate
        loss which is later used to adjust backpropagation for weight 
        correction during training. 

        Parameters:
        -------------------------------
        X : np.array
        - input sequence, sequence of several time
          dependent datapoints, often in the for of vectors (np.arrays) 
        
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
        time_steps = X.shape[0]
        h_layer_size = self.w_rec.shape[0]

        h_states = np.zeros(time_steps,h_layer_size)        
        
        z = X[t,:]@self.w_x + self.b_x
        h_t = self._act(z)
        h_states[0,:] = np.flatten(h_t)

        for t in range(1,time_steps):
            z = X[t,:]@self.w_x + h_states[t-1,:]@self.w_rec + self.b_rec
            h_t = self._act(z)
        
        h_states[t,:] = np.flatten(h_t)

        y_predicted = self.w_y@h_states + self.b_y #Kanskje h_states.T her

        return y_predicted #Predicted values for each hidden state/time
                           #step in a vector, meaning first element is 
                           #the networks prediction of what state two 
                           #will be given one input deom a sequence
                           # output two is then the prediction of the 
                           #next number in the sequence given two inputs
                           #from a sequence and so on
        

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
        
        input_size = X.shape[1]
        hidden_size = X.shape[0]
        output_size = y.shape[1]

        # weight input -> hidden.
        w_xh = np.random.randn(hidden_size, input_size)*0.01
        # weight hidden -> hidden     
        w_hh = np.random.randn(hidden_size, hidden_size)*0.01
        # weight hidden -> output
        w_hy = np.random.randn(output_size, hidden_size)*0.01

        #bias input -> hidden.
        b_x = np.random.randn(hidden_size, input_size)
        #bias hidden -> hidden.
        b_rec = np.random.randn(hidden_size, hidden_size)
        #bias hidden -> output.
        b_y = np.random.randn(output_size, hidden_size)

        #Do a forward pass, return outputs from each hidden state 
        #and all current weights and biases
        y_predicted = self._forward(X)
        #Find current loss from predicted output values
        loss = self._loss(y, y_predicted)

        for e in epochs:
            #Do a backprogation and adjust current weights and biases to 
            #improve loss, return new improved weights and biases
            self.w_xh,self.w_hh,self.w_hy,self.b_x,self.b_rec,self.b_y \
            = self._backward(loss)                                             #Maybe return weights and biases in two tuples of three instead??? Might be a lot cleaner                    
            #Do a forward pass with new weights, return outputs from 
            #each hidden state and all the weights and biases
            y_predicted = self._forward(X)
            #Predict the current loss with used weights and biases
            loss = self._loss(y, y_predicted)

            #Calculate improvement, the rate or amount of the network
            #improves per epoch
            improvement = None                                                 #INSERT gradient thingy or improvement measure here
            #Check if improvement is significant enough according to the
            #set threshold, if not break and do an early exit
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