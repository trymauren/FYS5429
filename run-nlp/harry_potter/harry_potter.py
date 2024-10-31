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
import multiprocessing as mp
from functools import partial

# ------ THIS FILE IS FOR INFERENCE ONLY :) ------ #

if __name__ == "__main__":

    # Adjust this to 4, 24 or 96
    seq_length = 96

    # This can be adjusted to any model in the
    # directory of the specified sequence length (above).
    # Text will be generated for each model with
    # hidden size in the list (they have to exist in the
    # subfolders of this directory in order for this to work):
    hidden_nodes_config = [5, 50, 500, 100]

    # Adjust this to the text you want to give as primer for further generation
    seed_str = 'Harry'

    loadpath = path_to_root + f'/run-nlp/harry_potter/saved_models/seq_length_{seq_length}/harry_potter'

    word_emb = WORD_EMBEDDING()

    """ Running inference for all models in hidden_node_config """
    for hidden_nodes in hidden_nodes_config:

        rnn = load_model(loadpath + f'_{hidden_nodes}_hidden')

        print(f'Generated text for a model with {hidden_nodes} hidden nodes:')
        X_seed = np.array([word_emb.get_embeddings(seed_str)])
        X_seed = X_seed.reshape(-1, 1, X_seed.shape[-1])
        ys = rnn.predict(X_seed, time_steps_to_generate=25,
                         return_seed_out=False, onehot=False)

        pred = f'[{seed_str}]'
        for emb in ys:
            pred += ' '
            pred += str(word_emb.find_closest(emb.flatten(), 1)[0])
            pred += ' '

        print(pred)

        plt.plot(rnn.get_stats()['loss'], label=f'Hidden size: {hidden_nodes}')

    plt.title('Loss over epochs for "Harry Potter"')
    plt.legend()
    plt.show()
