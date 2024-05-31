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
import tensorflow as tf
import multiprocessing as mp
from functools import partial

# ------ THIS FILE IS FOR INFERENCE :) ------ #

if __name__ == "__main__":

    seq_length = 24                                                                  # adjust this to 4, 24 or 96
    hidden_nodes_config = [400] # adjust this

    savepath = path_to_root + f'/run-nlp/harry_potter/saved_models/seq_length_{seq_length}/harry_potter'

    word_emb = WORD_EMBEDDING()

    for hidden_nodes in hidden_nodes_config:

        rnn = load_model(savepath + f'_{hidden_nodes}_hidden')

        print(f'Generated text for {hidden_nodes} hidden nodes:')
        seed_str = 'Harry did not do'
        X_seed = np.array([word_emb.get_embeddings(seed_str)])
        X_seed = X_seed.reshape(-1, 1, X_seed.shape[-1])
        ys = rnn.predict(X_seed, time_steps_to_generate=25,
                         return_seed_out=False, onehot=False)

        pred = '[Harry did not do]'
        for emb in ys:
            pred += ' '
            pred += str(word_emb.find_closest(emb.flatten(), 1)[0])
            pred += ' '

        print(pred)

        seed_str = 'never'
        X_seed = np.array([word_emb.get_embeddings(seed_str)])
        X_seed = X_seed.reshape(-1, 1, X_seed.shape[-1])
        ys = rnn.predict(X_seed, time_steps_to_generate=25,
                         return_seed_out=False, onehot=False)

        pred = '[never]'
        for emb in ys:
            pred += ' '
            pred += str(word_emb.find_closest(emb.flatten(), 1)[0])
            pred += ' '

        print(pred)

        plt.plot(rnn.get_stats()['loss'], label=f'Hidden size: {hidden_nodes}')

    plt.title('Loss over epochs for "Harry Potter"')
    plt.legend()
    plt.show()
