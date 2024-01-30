import numpy as np

class RecurrentNN:

    def __init__(self) -> None:
        pass

    def forward(self,X:np.array, w_x:np.array, w_rec:np.array) -> np.array:

        h_states = np.zeros((len(X)),self.n_hidden_nodes)        

        for i in range(len(X)):
            h_states[i,:] = X[i,:]@w_x + h_states[i-1,:]@w_rec
        

    def _backward(self, y_estimate : np.ndarray) -> None:
        pass

    def fit(self,
            X : np.ndarray,
            y : np.ndarray,
            epochs : int,
            # early_stopping_params,
            optimization_algorithm : str = None
        ) -> None:
        
        w_hh= np.zeros(self.n_hidden_nodes, self.n_hidden_nodes) #State values for each node in each state

    def predict(self, X : np.ndarray) -> np.ndarray:
        pass