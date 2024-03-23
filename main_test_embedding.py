import sys
import git
import numpy as np
from rnn.rnn import RNN
from utils.activations import Relu, Tanh
import matplotlib.pyplot as plt
import utils.text_processing as text_proc
from utils.text_processing import WORD_EMBEDDING
path_to_root = git.Repo('.', search_parent_directories=True).working_dir
sys.path.append(path_to_root)

seq_length = 6
examples = 50
epo = 1500
hidden_nodes = 600
rnn = RNN(
    hidden_activation='Tanh()',
    output_activation='Tanh()',
    loss_function='mse()',
    optimiser='AdaGrad()',
    regression=True)

word_emb = WORD_EMBEDDING()

X = np.array(
    [word_emb.get_embeddings(text_proc.read_file("utils/embedding_test.txt"))]
    )
print(X.shape)
y = np.array(
    [word_emb.get_embeddings(text_proc.read_file("utils/embedding_test_y.txt"))]
    )

whole_sequence_output, hidden_state = rnn.fit(
    X, y, epo, learning_rate=0.002,
    num_hidden_nodes=hidden_nodes, return_sequences=True)

plt.plot(rnn.get_stats()['loss'])
plt.show()
x_seed = X[0][0]
print(word_emb.find_closest(x_seed, number=1))
ret = rnn.predict(x_seed, hidden_state, 10)

for emb in ret:
    word = word_emb.find_closest(emb,number=3)
    print(word)
plt.plot(ret)
plt.show()

