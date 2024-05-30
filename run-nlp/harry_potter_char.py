import sys
import git
path_to_root = git.Repo('.', search_parent_directories=True).working_dir
sys.path.append(path_to_root)
import numpy as np
import matplotlib.pyplot as plt
from rnn.rnn import RNN as RNN_parallel
import utils.text_processing as text_proc
from utils.text_processing import WORD_EMBEDDING
from utils.read_load_model import load_model
import resource


epo = 2000
hidden_nodes = 400
unrolling_steps = 25
learning_rate = 0.001
optimiser = 'Adam()'
num_batches = 1

text_data = text_proc.read_file(path_to_root + '/data/embedding_test.txt')
chars = sorted(list(set(text_data)))  # to keep the order consistent over runs
data_size, vocab_size = len(text_data), len(chars)
print(f'Size: {data_size}, unique: {vocab_size}.')
char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for i, ch in enumerate(chars)}

# # word_emb = WORD_EMBEDDING()
# text_data = text_proc.read_file(path_to_root + '/data/harry_potter.txt')
# # text_data = text_data[:100]
# chars = list(set(text_data))
# data_size, vocab_size = len(text_data), len(chars)
# print('data has %d characters, %d unique.' % (data_size, vocab_size))
# char_to_ix = {ch: i for i, ch in enumerate(chars)}
# ix_to_char = {i: ch for i, ch in enumerate(chars)}

seq_length = 25  # just 2^somenumber

inputs = [char_to_ix[ch] for ch in text_data[0:seq_length]]
targets = [char_to_ix[ch] for ch in text_data[1:seq_length+1]]

X = text_proc.create_onehot(text_data[0:seq_length], char_to_ix)
y = text_proc.create_onehot(text_data[1:seq_length+1], char_to_ix)

X = np.array([X])
y = np.array([y])

X = X.reshape(1, -1, num_batches, vocab_size)
y = y.reshape(1, -1, num_batches, vocab_size)

print('Shape of X after batching:', X.shape)
print('Shape of y after batching:', y.shape)

savepath = path_to_root + '/run-nlp/saved_models/onehot_test'
train = True
train = False

if train:
    rnn = RNN_parallel(
        hidden_activation='Tanh()',
        output_activation='Softmax()',
        loss_function='ce()',
        optimiser=optimiser,
        clip_threshold=1,
        learning_rate=learning_rate,
        name=savepath,
        seed=23,
        )

    hidden_state_batch = rnn.fit(
        X,
        y,
        epo,
        num_hidden_nodes=hidden_nodes,
        unrolling_steps=unrolling_steps,
        vocab=ix_to_char,
        # num_epochs_no_update=200,
        # gradcheck_at=3,
    )

bytes_usage_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
gb_usage_peak = round(bytes_usage_peak/1000000000, 3)
print('Memory consumption (peak):')
print(gb_usage_peak, 'GB')


rnn = load_model(savepath)
seed_str = 'bi'

X_seed = text_proc.create_onehot(seed_str, char_to_ix)

ys = rnn.predict(X_seed, time_steps_to_generate=50,
                 return_seed_out=False, onehot=True)  # change seed out to True

pred = ''
for y in ys:
    ix = text_proc.onehot_to_ix(y)
    pred += ix_to_char[ix]

print(pred)

plt.plot(rnn.get_stats()['loss'], label='batch train')
plt.legend()
plt.show()
