import numpy as np

class RecurrentNN:

    def __init__(self) -> None:
        pass

    def forward(self,X:np.array, W_x:np.array, W_rec:np.array) -> np.array:
        
        time_steps = X.shape[0]
        h_layer_size = W_rec.shape[0]

        h_states = np.zeros(time_steps,h_layer_size)        
        
        z = X[t,:]@W_x
        h_t = self.act(z)
        h_states[0,:] = np.flatten(h_t)

        for t in range(1,time_steps):
            z = X[t,:]@W_x + h_states[t-1,:]@W_rec
            h_t = self.act(z)
        
        h_states[t,:] = np.flatten(h_t)

        

    def _backward(self, y_estimate : np.ndarray) -> None:
        pass

    def fit(self,
            X : np.ndarray,
            y : np.ndarray,
            epochs : int,
            # early_stopping_params,
            optimization_algorithm : str = None
        ) -> None:
        
        W_xh = np.random.randn(hidden_size, num_chars)*0.01     # weight input -> hidden. 
        W_hh = np.random.randn(hidden_size, hidden_size)*0.01   # weight hidden -> hidden
        W_hy = np.random.randn(num_chars, hidden_size)*0.01     # weight hidden -> output

    def predict(self, X : np.ndarray) -> np.ndarray:
        pass