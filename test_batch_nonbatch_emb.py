import sys
import git
import numpy as np
import matplotlib.pyplot as plt
from rnn.rnn import RNN as RNN
from rnn.rnn_batch_new import RNN as RNN_parallel
import utils.text_processing as text_proc
from utils.text_processing import WORD_EMBEDDING
path_to_root = git.Repo('.', search_parent_directories=True).working_dir
sys.path.append(path_to_root)

epo = 100
hidden_nodes = 30
num_backsteps = 30
num_forwardsteps = 30
learning_rate = 0.001
optimiser = 'AdaGrad()'
num_batches = 2

word_emb = WORD_EMBEDDING()
text_data = text_proc.read_file("data/three_little_pigs.txt")
X, y = np.array(word_emb.translate_and_shift(text_data))
X = np.array([X])
y = np.array([y])
vocab, inverse_vocab = text_proc.create_vocabulary(X)
y = text_proc.create_labels(X, inverse_vocab)
X = X.reshape(1, -1, num_batches, X.shape[-1])
X = X[:, 0:15, :, :]
y = y.reshape(1, -1, num_batches, y.shape[-1])
y = y[:, 0:15, :, :]
print('X:', X.shape)
print('y:', y.shape)

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
    loss_function='ce()',
    optimiser=optimiser,
    clip_threshold=1,
    learning_rate=learning_rate,
    )

hidden_state_batch = rnn_batch.fit(
    X,
    y,
    epo,
    num_hidden_nodes=hidden_nodes,
    num_backsteps=num_backsteps,
    num_forwardsteps=num_forwardsteps,
    gradcheck_at=3,
)

plt.plot(rnn_batch.get_stats()['loss'], label='batch train')
plt.plot(rnn_batch.get_stats()['val_loss'], label='batch val')

plt.legend()
plt.show()
