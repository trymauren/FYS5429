import numpy as np

class RecurrentNN:

    def __init__(self) -> None:
        pass

    def _forward(self, X : np.ndarray) -> np.ndarray:
        pass

    def _backward(self, y_estimate : np.ndarray) -> None:
        pass

    def fit(self,
            X : np.ndarray,
            y : np.ndarray,
            epochs : int,
            # early_stopping_params,
            optimization_algorithm : str = None
        ) -> None:
        
        pass

    def predict(self, X : np.ndarray) -> np.ndarray:
        pass