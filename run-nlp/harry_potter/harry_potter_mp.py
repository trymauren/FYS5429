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
import multiprocessing as mp
from functools import partial


# ------ THIS FILE IS FOR RUNNING EXPERIMENTS IN PARALLEL ------ #

seq_length = 24
num_sequences = 500
savepath = path_to_root + f'/run-nlp/harry_potter/saved_models/seq_length_{seq_length}/harry_potter'
train = False
infer = True


def train_rnn(hidden_nodes, savepath=None,
              X=None, y=None, vocab=None):

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
        vocab=vocab
    )


if __name__ == "__main__":

    word_emb = WORD_EMBEDDING()
    text_data = text_proc.read_file(path_to_root + '/data/harry_potter.txt')

    X, y = np.array(word_emb.translate_and_shift(text_data))
    X = np.array([X])

    vocab, inverse_vocab = text_proc.create_vocabulary(X)
    y = text_proc.create_labels(X, inverse_vocab)

    inputs = np.zeros((1, num_sequences, seq_length, X.shape[-1]))
    targets = np.zeros((1, num_sequences, seq_length, y.shape[-1]))
    print(inputs.shape)
    print(targets.shape)

    for i in range(num_sequences):
        input_ = X[0, i:i + seq_length, :]
        target = y[0, i + 1:i+seq_length+1, :]
        inputs[0, i] = input_
        targets[0, i] = target

    X = inputs.transpose(0, 2, 1, 3)
    y = targets.transpose(0, 2, 1, 3)

    print('Shape of X after batching:', X.shape)
    print('Shape of y after batching:', y.shape)

    hidden_nodes_config = [1, 2, 3, 4, 5, 10, 20, 50, 100, 200, 300, 400, 500, 1000]

    partial_train_rnn = partial(train_rnn,
                                savepath=savepath, X=X, y=y,
                                vocab=vocab)

    if train:

        with mp.Pool(processes=len(hidden_nodes_config)) as pool:
            pool.map(partial_train_rnn, hidden_nodes_config)

    if infer:

        for hidden_nodes in hidden_nodes_config:

            rnn = load_model(savepath + f'_{hidden_nodes}_hidden')

            print(f'Generated text for {hidden_nodes} hidden nodes:')
            seed_str = 'Harry was'
            X_seed = np.array([word_emb.get_embeddings(seed_str)])
            X_seed = X_seed.reshape(-1, 1, X_seed.shape[-1])
            ys = rnn.predict(X_seed, time_steps_to_generate=25,
                             return_seed_out=False, onehot=False)

            pred = '[Harry was]'
            for emb in ys:
                pred += ' '
                pred += str(word_emb.find_closest(emb.flatten(), 1)[0])
                pred += ' '

            print(pred)

            seed_str = 'They were'
            X_seed = np.array([word_emb.get_embeddings(seed_str)])
            X_seed = X_seed.reshape(-1, 1, X_seed.shape[-1])
            ys = rnn.predict(X_seed, time_steps_to_generate=25,
                             return_seed_out=False, onehot=False)

            pred = '[They were]'
            for emb in ys:
                pred += ' '
                pred += str(word_emb.find_closest(emb.flatten(), 1)[0])
                pred += ' '

            print(pred)

            plt.plot(rnn.get_stats()['loss'], label=f'Hidden size: {hidden_nodes}')

        plt.title('Loss over epochs for "Harry Potter"')
        plt.legend()
        plt.savefig(path_to_root + f'/run-nlp/harry_potter/saved_figs/seq_length_{seq_length}_loss.svg')
