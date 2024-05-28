import sys
import git
import numpy as np
import matplotlib.pyplot as plt
from rnn.rnn import RNN as RNN_parallel
import utils.text_processing as text_proc
from utils.text_processing import WORD_EMBEDDING
from utils.read_load_model import load_model
import tensorflow as tf
import resource

path_to_root = git.Repo('.', search_parent_directories=True).working_dir
sys.path.append(path_to_root)

epo = 1000
hidden_nodes = 1200
unrolling_steps = 50
learning_rate = 0.01
optimiser = 'Adam()'
num_batches = 32

word_emb = WORD_EMBEDDING()
text_data = text_proc.read_file("data/harry_potter.txt")

# path_to_file = tf.keras.utils.get_file('shakespeare.txt','https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
# text_data = open(path_to_file, 'rb').read().decode(encoding='utf-8')[:100000]

X, y = np.array(word_emb.translate_and_shift(text_data))
X = np.array([X])
y = np.array([y])

# picking out a number of words (9520) that can be divided by batchsize=16
X = X[:, :9536, :]
y = y[:, :9536, :]
print('Shape of X after picking first 9520 words', X.shape)
print('Shape of y after picking first 9520 words', y.shape)

vocab, inverse_vocab = text_proc.create_vocabulary(X)
y = text_proc.create_labels(X, inverse_vocab)
print('Shape of X after onehot-encoding of y:', X.shape)
print('Shape of y after onehot-encoding of y:', y.shape)

X = X.reshape(1, -1, num_batches, X.shape[-1])
y = y.reshape(1, -1, num_batches, y.shape[-1])
print('Shape of X after batching:', X.shape)
print('Shape of y after batching:', y.shape)


train = True
train = True
if train:
    rnn_batch = RNN_parallel(
        hidden_activation='Tanh()',
        output_activation='Softmax()',
        loss_function='ce()',
        optimiser=optimiser,
        clip_threshold=1,
        learning_rate=learning_rate,
        name='saved_models/harry_potter_fox_test'
        )

    hidden_state_batch = rnn_batch.fit(
        X,
        y,
        epo,
        num_hidden_nodes=hidden_nodes,
        unrolling_steps=unrolling_steps,
        vocab=vocab,
        # num_epochs_no_update=200,
        # gradcheck_at=3,
    )

bytes_usage_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
gb_usage_peak = round(bytes_usage_peak/1000000000, 3)
print('Memory consumption (peak):')
print(gb_usage_peak, 'GB')

rnn_batch = load_model('saved_models/harry_potter_test_2')
seed_str = 'Harry potter'
X_seed = np.array([word_emb.get_embeddings(seed_str)])
ys, seed_out = rnn_batch.predict(X_seed.reshape(-1, 1, X_seed.shape[-1]), time_steps_to_generate=10, return_seed_out=True)

pre_str = ''
for emb, seed in zip(seed_out, seed_str.split()):
    pre_str += seed
    pre_str += ' '
    pre_str += str(word_emb.find_closest(emb.flatten(), 1))
    pre_str += ' '

generated_string = '|'
for emb in ys:
    generated_string += ' '
    generated_string += str(word_emb.find_closest(emb.flatten(), 1))

print(pre_str + generated_string)


# plt.plot(rnn_batch.get_stats()['loss'], label='batch train')
# plt.legend()
# plt.show()

