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
import tensorflow as tf  # only for dataset

# ------ THIS FILE IS FOR INFERENCE :) ------ #

if __name__ == "__main__":

    savepath = path_to_root + '/run-nlp/romeo_and_juliet/saved_models/seq_length_24/romeo_and_juliet'

    path_to_file = tf.keras.utils.get_file('shakespeare.txt','https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
    text_data = open(path_to_file, 'rb').read().decode(encoding='utf-8')[:100000]
    chars = sorted(list(set(text_data)))  # to keep the order consistent over runs
    data_size, vocab_size = len(text_data), len(chars)
    print(f'Size: {data_size}, unique: {vocab_size}.')
    char_to_ix = {ch: i for i, ch in enumerate(chars)}
    ix_to_char = {i: ch for i, ch in enumerate(chars)}

    hidden_nodes_config = [1, 2, 3, 4, 5, 10, 20, 100, 200]

    for hidden_nodes in hidden_nodes_config:

        rnn = load_model(savepath + f'_{hidden_nodes}_hidden')

        plt.plot(rnn.get_stats()['loss'], label=f'Hidden size: {hidden_nodes}')

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

    plt.title('Loss over epochs for "Romeo and Juliet"')
    plt.legend()
    plt.show()
