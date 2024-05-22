import sys
import git
import numpy as np
import matplotlib.pyplot as plt
from rnn.rnn import RNN as RNN
from rnn.rnn_batch_new import RNN as RNN_parallel
path_to_root = git.Repo('.', search_parent_directories=True).working_dir
sys.path.append(path_to_root)


def create_sines(examples=10, seq_length=100):
    X = []
    y = []
    for _ in range(examples):
        example_x = np.array(
            [np.sin(
                np.linspace(0, 8*np.pi, seq_length+1))]
            ).T
        # example_x = np.repeat(example_x, 2, axis=1)
        X.append(example_x[0:-1])
        y.append(example_x[1:])

    return np.array(X, dtype=np.float64), np.array(y, dtype=np.float64)


seq_length = 30
examples = 1
epo = 1000
hidden_nodes = 4
num_backsteps = 15
learning_rate = 0.001
optimiser = 'AdaGrad()'
num_batches = 2
features = 1

X, y = create_sines(examples=examples, seq_length=seq_length)

X_batched = X.reshape(examples, -1, num_batches, features)
y_batched = y.reshape(examples, -1, num_batches, features)
X_nonbatched = X.reshape(examples, num_batches, -1, features)
y_nonbatched = y.reshape(examples, num_batches, -1, features)


# rnn = RNN(
#     hidden_activation='Tanh()',
#     output_activation='Identity()',
#     loss_function='mse()',
#     optimiser=optimiser,
#     clip_threshold=1,
#     learning_rate=learning_rate,
#     )

# hidden_state = rnn.fit(
#     X_nonbatched,
#     y_nonbatched,
#     epo,
#     num_hidden_nodes=hidden_nodes,
#     num_backsteps=num_backsteps,
#     num_forwardsteps=num_backsteps,
# )
# plt.plot(rnn.get_stats()['loss'], label='rnn')



rnn_batch = RNN_parallel(
    hidden_activation='Tanh()',
    output_activation='Identity()',
    loss_function='mse()',
    optimiser=optimiser,
    clip_threshold=1,
    learning_rate=learning_rate,
    )

hidden_state_batch = rnn_batch.fit(
    X_batched,
    y_batched,
    epo,
    num_hidden_nodes=hidden_nodes,
    num_backsteps=num_backsteps,
    num_forwardsteps=num_backsteps,
)

# plt.plot(rnn_batch.get_stats()['loss'], label='batch')

# plt.legend()
# plt.show()