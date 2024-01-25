import utils
import autograd.numpy as np

class NeuralNetwork:

    def __init__(self, hidden_layer_sizes: list[int], alpha: int, solver: str):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha = alpha
        self.solver = solver

        self.weights = None
        self.biases = None

        self.X = None

    def _forward(self) -> np.array: 
        z = 0
        count = 0
        for x in self.X:
            z = x
            for W,b in zip(self.weights, self.biases):
                Wx = np.dot(z,W.T) + b
                z = utils.ReLU(Wx)
        return z

    def _backward(self, y_pred, y_true):
        # this is the formula for log-loss (cross-entropy). Need to fit it to our case
        # and it may also may not be right to use this for regression. Consider MSE?
        return -(y*log(p) + (1-y) * log(1-p))

    def fit(self, X, y, epochs, batch=False):

        if len(X.shape) == 1: # snatched :o
            X = X.reshape((1, X.shape[0]))

        self.X = X

        self.input_size = X.shape[1]
        self.output_size = y.shape[0]

        # Merging to one list
        layer_sizes = [self.input_size] + self.hidden_layer_sizes + [self.output_size] 

        self.weights = utils.init_weights(layer_sizes, random_weight=0.01)
        self.biases = utils.init_biases(layer_sizes, random_weight=0.01)

        # if batch:
        #     # Should implement some kind of batch-processing?
        #     pass

        y_out = self._forward()

        for e in range(epochs):
            # self._backward()
            y_out = self._forward()


        return y_out

    def predict(self, X, classify:bool=False, threshold:float=0.5):
        if len(X.shape) == 1: # snatched :o
            X = X.reshape((1, X.shape[0]))

        self.X = X # OVERSKRIVER TRENINGS-X. OK?
        y_pred = self._forward()
        
        if classify:
            y_pred = np.array([1 * (n > threshold) for n in y_pred])

        return y_pred




nn = NeuralNetwork([10,10], 0.05, 'id')

X = np.array([[0,1],
              [2,3],
              [3,1],
              [2,3],
              [4,2]])

y = np.array([1,5,4,5,6]).T
f = nn.fit(X,y,100)
g = nn.predict(X, classify=True)
print(g)