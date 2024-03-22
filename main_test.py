import sys
import git
import numpy as np
from rnn.rnn import RNN
from utils.activations import Relu, Tanh
import matplotlib.pyplot as plt
path_to_root = git.Repo('.', search_parent_directories=True).working_dir
sys.path.append(path_to_root)



def create_sines(examples=10, seq_length=100):
    X = []
    y = []
    for _ in range(examples):
        noise_x = np.array([np.random.normal(0, 1, seq_length)])
        example_x = np.array([np.sin(np.linspace(0, 8 * np.pi, seq_length))])
        example_w_noise_x = (example_x + noise_x).T
        example_w_noise_y = (example_x + noise_x).T

        X.append(example_w_noise_x)
        y.append(example_w_noise_y)
    return X, y


seq_length = 100
examples = 20
epo = 1000
hidden_nodes = 70
rnn = RNN(
    hidden_activation='Tanh()',
    output_activation='Tanh()',
    loss_function='mse()',
    optimiser='AdaGrad()',
    regression=True)

X, y = create_sines(examples=examples, seq_length=seq_length)
# X_val, y_val = create_sines(examples=1, seq_length=seq_length)
ret = rnn.fit(X, y, epo, learning_rate=0.1, num_hidden_states=seq_length, num_hidden_nodes=hidden_nodes)
# X, y = create_sines(examples=examples, seq_length=50)
# rnn.fit(X, y, epo, num_hidden_states=50, num_hidden_nodes=hidden_nodes)
# y_pred_val = rnn.predict(X)
# plt.plot(X_val[0], label='X_val', linestyle='dotted')
# plt.plot(y_pred_val, label='y_pred_val', linestyle='dashdot')
# plt.plot(y_val[0], label='y_cal')
# plt.legend()
# plt.show()
# rnn.predict()

stats = rnn.get_stats()
plt.plot(stats['loss'], label='loss')
plt.legend()
plt.show()
# plt.plot(ret, label='output sine')
# plt.plot(y, label='true sine')
# plt.legend()
# plt.show()
