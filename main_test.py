import sys
import git
import numpy as np
from rnn.rnn import RNN
from utils.activations import Relu, Tanh
import matplotlib.pyplot as plt
import utils.text_processing as text_proc
from utils.text_processing import WORD_EMBEDDING
path_to_root = git.Repo('.', search_parent_directories=True).working_dir
sys.path.append(path_to_root)


def create_sines(examples=10, seq_length=100):
    X = []
    y = []
    for _ in range(examples):
        noise_x = np.array([np.random.normal(0, 1, seq_length)])
        example_x = np.array([np.sin(np.linspace(0, 8 * np.pi, seq_length))])
        example_w_noise_x = (example_x + noise_x).T
        # example_w_noise_y = (example_x + noise_x).T
        # X.append(example_w_noise_x)
        X.append(example_x.T)
        # example_x[0][0] = 0
        # example_x[0][1] = 0
        # example_x[0][2] = 0
        # example_x[0][3] = 0
        # example_x[0][4] = 0
        # example_x[0][5] = 0
        # example_x[0][6] = 0
        # example_x[0][7] = 0
        y.append(example_x.T)
        # y.append(example_w_noise_y)
    return X, y


seq_length = 20
examples = 50
epo = 10000
hidden_nodes = 50
rnn = RNN(
    hidden_activation='Tanh()',
    output_activation='Tanh()',
    loss_function='mse()',
    optimiser='AdaGrad()',
    regression=True)

X, y = create_sines(examples=examples, seq_length=seq_length)
X_val, y_val = create_sines(examples=1, seq_length=seq_length)
whole_sequence_output, hidden_state = rnn.fit(
    X, y, epo, learning_rate=0.0001, num_hidden_states=seq_length,
    num_hidden_nodes=hidden_nodes, return_sequences=True)

plt.plot(rnn.get_stats()['loss'])
plt.show()
x_seed = np.array([1])
ret = rnn.predict(x_seed, hidden_state, 4)
plt.plot(ret)
plt.show()
# X, y = create_sines(examples=examples, seq_length=50)
# rnn.fit(X, y, epo, num_hidden_states=50, num_hidden_nodes=hidden_nodes)
# y_pred_val = rnn.predict(X_val)
# plt.plot(X_val[0], label='X_val', linestyle='solid', color='red')
# plt.plot(y_pred_val, label='y_pred_val', linestyle='dotted', color='blue')
# plt.plot(y_val[0], label='y_val')
# plt.legend()
# plt.show()

# stats = rnn.get_stats()
# plt.plot(stats['loss'], label='loss')
# plt.legend()
# plt.show()
# plt.plot(ret, label='output sine')
# plt.plot(y, label='true sine')
# plt.legend()
# plt.show()
