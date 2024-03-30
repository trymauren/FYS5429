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
        #noise_x = np.array([np.random.normal(0, 1, seq_length)])
        example_x = np.array([np.sin(np.linspace(0, 8 * np.pi, seq_length))])
        #example_w_noise_x = (example_x + noise_x).T
        # example_w_noise_y = (example_x + noise_x).T
        # X.append(example_w_noise_x)
        X.append(example_x[0:-1].T)
        # example_x[0][0] = 0
        # example_x[0][1] = 0
        # example_x[0][2] = 0
        # example_x[0][3] = 0
        # example_x[0][4] = 0
        # example_x[0][5] = 0
        # example_x[0][6] = 0
        # example_x[0][7] = 0
        y.append(example_x[1:].T)
        # y.append(example_w_noise_y)
    return X, y


seq_length = 20
examples = 50
epo = 200
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

X_seed = X_val
ret = rnn.predict(X[0][0], hidden_state, 4)
print(ret)
plt.plot(ret)
plt.show()

epo = 1000
hidden_nodes = 300
learning_rates = [0.001, 0.003, 0.005, 0.01]

for learning_rate_curr in learning_rates:
    fig, ax = plt.subplots()
    print(f'learning rate: {learning_rate_curr}')
    rnn = RNN(
        hidden_activation='Tanh()',
        output_activation='Identity()',
        loss_function='mse()',
        optimiser='AdaGrad()',
        regression=True,
        threshold=1,
        )

    whole_sequence_output, hidden_state = rnn.fit(
        X, y, epo,
        num_hidden_nodes=hidden_nodes, return_sequences=True,
        independent_samples=True, learning_rate=learning_rate_curr)

    rnn.plot_loss(plt, figax=(fig, ax), show=False)

    predict = rnn.predict(X_seed)
    plt.plot(predict)

    rnn = RNN(
        hidden_activation='Tanh()',
        output_activation='Identity()',
        loss_function='mse()',
        optimiser='SGD()',
        regression=True,
        threshold=1,
        )

    whole_sequence_output, hidden_state = rnn.fit(
        X, y, epo,
        num_hidden_nodes=hidden_nodes, return_sequences=True,
        independent_samples=True, learning_rate=learning_rate_curr
        )

    rnn.plot_loss(plt, figax=(fig, ax), show=False)

    predict = rnn.predict(X_seed)
    plt.plot(predict)

    rnn = RNN(
        hidden_activation='Tanh()',
        output_activation='Identity()',
        loss_function='mse()',
        optimiser='SGD_momentum()',
        regression=True,
        threshold=1,
        )

    whole_sequence_output, hidden_state = rnn.fit(
        X, y, epo,
        num_hidden_nodes=hidden_nodes, return_sequences=True,
        independent_samples=True, learning_rate=learning_rate_curr,
        momentum_rate=0.9)

    rnn.plot_loss(plt, figax=(fig, ax), show=False)

    predict = rnn.predict(X_seed)
    plt.plot(predict)

    rnn = RNN(
        hidden_activation='Tanh()',
        output_activation='Identity()',
        loss_function='mse()',
        optimiser='RMSProp()',
        regression=True,
        threshold=1,
        )

    whole_sequence_output, hidden_state = rnn.fit(
        X, y, epo,
        num_hidden_nodes=hidden_nodes, return_sequences=True,
        independent_samples=True, learning_rate=learning_rate_curr,
        decay_rate=0.001)

    rnn.plot_loss(plt, figax=(fig, ax), show=True)

    predict = rnn.predict(X_seed)
    plt.plot(predict)




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
