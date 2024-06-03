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

# ------ THIS FILE IS FOR INFERENCE ONLY :) ------ #

if __name__ == "__main__":

    # Adjust this to 4, 24 or 96
    seq_length = 24

    # This can be adjusted to any model in the
    # directory of the specified sequence length (above).
    # Text will be generated for each model with
    # hidden size in the list:
    hidden_nodes_config = [4, 50, 400, 100]

    # Adjust this to the text you want to give as primer for further generation
    seed_str_1 = 'ROMEO'
    seed_str_2 = 'JULIET'

    loadpath = path_to_root + f'/run-nlp/romeo_and_juliet/saved_models/seq_length_{seq_length}/romeo_and_juliet'

    """ Reading text and creating vocabulary """
    path_to_file = tf.keras.utils.get_file('shakespeare.txt','https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
    text_data = open(path_to_file, 'rb').read().decode(encoding='utf-8')[:100000]
    chars = sorted(list(set(text_data)))  # to keep the order consistent over runs
    data_size, vocab_size = len(text_data), len(chars)
    print(f'Size: {data_size}, unique: {vocab_size}.')
    char_to_ix = {ch: i for i, ch in enumerate(chars)}
    ix_to_char = {i: ch for i, ch in enumerate(chars)}

    """ Running inference for all models in hidden_node_config """
    for hidden_nodes in hidden_nodes_config:

        rnn = load_model(loadpath + f'_{hidden_nodes}_hidden')

        print(f'Generated text for {hidden_nodes} hidden nodes:')

        ixs = [char_to_ix[ch] for ch in seed_str_1]
        X_seed = text_proc.create_onehot(ixs, char_to_ix)
        ys = rnn.predict(X_seed, time_steps_to_generate=25,
                         return_seed_out=False, onehot=True)

        pred = f'[{seed_str_1}]'
        for char in ys:
            ix = text_proc.onehot_to_ix(char)
            pred += ix_to_char[ix]

        print(pred)

        ixs = [char_to_ix[ch] for ch in seed_str_2]
        X_seed = text_proc.create_onehot(ixs, char_to_ix)
        ys = rnn.predict(X_seed, time_steps_to_generate=25,
                         return_seed_out=False, onehot=True)

        pred = f'[{seed_str_2}]'
        for char in ys:
            ix = text_proc.onehot_to_ix(char)
            pred += ix_to_char[ix]

        print(pred)

        plt.plot(rnn.get_stats()['loss'], label=f'Hidden size: {hidden_nodes}')

    plt.title('Loss over epochs for "Romeo and Juliet"')
    plt.legend()
    plt.show()
