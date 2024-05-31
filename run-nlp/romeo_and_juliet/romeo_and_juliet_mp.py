import sys
import git
path_to_root = git.Repo('.', search_parent_directories=True).working_dir
sys.path.append(path_to_root)
import numpy as np
import matplotlib.pyplot as plt
from rnn.rnn import RNN
import utils.text_processing as text_proc
from utils.text_processing import WORD_EMBEDDING
from utils.read_load_model import load_model
import resource
import tensorflow as tf  # only for dataset
import multiprocessing as mp
from functools import partial

seq_length = 24
num_sequences = 5000
savepath = path_to_root + f'/run-nlp/romeo_and_juliet/saved_models/seq_length_{seq_length}/romeo_and_juliet'
train = False
infer = True


def train_rnn(hidden_nodes, savepath=None,
              X=None, y=None, ix_to_char=None):

    epo = 1000
    learning_rate = 0.001
    optimiser = 'Adam()'

    rnn = RNN(
        hidden_activation='Tanh()',
        output_activation='Softmax()',
        loss_function='ce()',
        optimiser=optimiser,
        clip_threshold=1,
        learning_rate=learning_rate,
        name=savepath + f'_{hidden_nodes}_hidden',
        seed=23
        )

    rnn.fit(
        X,
        y,
        epo,
        num_hidden_nodes=hidden_nodes,
        vocab=ix_to_char
    )


if __name__ == "__main__":
    path_to_file = tf.keras.utils.get_file('shakespeare.txt','https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
    text_data = open(path_to_file, 'rb').read().decode(encoding='utf-8')[:100000]
    chars = sorted(list(set(text_data)))  # to keep the order consistent over runs
    data_size, vocab_size = len(text_data), len(chars)
    print(f'Size: {data_size}, unique: {vocab_size}.')
    char_to_ix = {ch: i for i, ch in enumerate(chars)}
    ix_to_char = {i: ch for i, ch in enumerate(chars)}

    X = np.zeros((num_sequences, seq_length, 1, len(char_to_ix)))
    y = np.zeros((num_sequences, seq_length, 1, len(char_to_ix)))

    for i in range(num_sequences):
        inputs = [char_to_ix[ch] for ch in text_data[i:i + seq_length]]
        targets = [char_to_ix[ch] for ch in text_data[i + 1:i+seq_length+1]]
        onehot_x = text_proc.create_onehot(inputs, char_to_ix)
        onehot_y = text_proc.create_onehot(targets, char_to_ix)
        X[i] = onehot_x
        y[i] = onehot_y

    X = X.transpose((2, 1, 0, 3))
    y = y.transpose((2, 1, 0, 3))

    print('Shape of X after batching:', X.shape)
    print('Shape of y after batching:', y.shape)

    partial_train_rnn = partial(train_rnn,
                                savepath=savepath, X=X, y=y,
                                ix_to_char=ix_to_char)

    hidden_nodes_config = [1, 2, 3, 4, 5, 10, 20, 50, 100, 200, 300, 400, 500, 1000]

    if train:

        with mp.Pool(processes=len(hidden_nodes_config)) as pool:
            pool.map(partial_train_rnn, hidden_nodes_config)

    if infer:

        for hidden_nodes in hidden_nodes_config:

            rnn = load_model(savepath + f'_{hidden_nodes}_hidden')

            print(f'Generated text for {hidden_nodes} hidden nodes:')
            seed_str = 'ROMEO'
            ixs = [char_to_ix[ch] for ch in seed_str]
            X_seed = text_proc.create_onehot(ixs, char_to_ix)
            ys = rnn.predict(X_seed, time_steps_to_generate=25,
                             return_seed_out=False, onehot=True)

            pred = '[ROMEO]'
            for char in ys:
                ix = text_proc.onehot_to_ix(char)
                pred += ix_to_char[ix]

            print(pred)

            seed_str = 'JULIET'
            ixs = [char_to_ix[ch] for ch in seed_str]
            X_seed = text_proc.create_onehot(ixs, char_to_ix)
            ys = rnn.predict(X_seed, time_steps_to_generate=25,
                             return_seed_out=False, onehot=True)

            pred = '[JULIET]'
            for char in ys:
                ix = text_proc.onehot_to_ix(char)
                pred += ix_to_char[ix]

            print(pred)

            plt.plot(rnn.get_stats()['loss'], label=f'Hidden size: {hidden_nodes}')

        plt.title('Loss over epochs for "Romeo and Juliet"')
        plt.legend()
        plt.show()
        # plt.savefig(path_to_root + f'/run-nlp/romeo_and_juliet/saved_figs/romeo_and_juliet_seq_length_24_all_hidden.svg')
