import sys
import git
import numpy as np
from rnn.rnn import RNN
import matplotlib.pyplot as plt
import utils.text_processing as text_proc
from utils.text_processing import WORD_EMBEDDING
path_to_root = git.Repo('.', search_parent_directories=True).working_dir
sys.path.append(path_to_root)

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)


word_emb = WORD_EMBEDDING()

text_data = text_proc.read_file("utils/three_little_pigs.txt")

X,y = np.array(word_emb.translate_and_shift(text_data))
print(X.shape)
X = np.array([X])
y = np.array([y])

X_seed = np.array([word_emb.get_embeddings("three")])

epo = 50
hidden_nodes = 100
learning_rates = [0.001, 0.003, 0.005, 0.01]

rnn = RNN(
    hidden_activation='Tanh()',
    output_activation='Identity()',
    loss_function='mse()',
    optimiser='AdaGrad()',
    clip_threshold=1,
    )

hidden_state = rnn.fit(
    X, y, epo,
    num_hidden_nodes=hidden_nodes, return_sequences=True,
    independent_samples=False, learning_rate=0.005, num_backsteps=100)
rnn.plot_loss(plt, show=True)

predict = rnn.predict(X_seed, time_steps_to_generate=3)

# with open("child_book_1_dump.pkl", "wb") as f:
#     pickle.dump(predict, f)

for emb in predict:
    print(word_emb.find_closest(emb, 1))

#for learning_rate_curr in learning_rates:
#    fig, ax = plt.subplots()
#    print(f'learning rate: {learning_rate_curr}')
#    rnn = RNN(
#        hidden_activation='Tanh()',
#        output_activation='Identity()',
#        loss_function='mse()',
#        optimiser='AdaGrad()',
#        regression=True,
#        threshold=1,
#        )
#
#    whole_sequence_output, hidden_state = rnn.fit(
#        X, y, epo,
#        num_hidden_nodes=hidden_nodes, return_sequences=True,
#        independent_samples=True, learning_rate=learning_rate_curr)
#
#    rnn.plot_loss(plt, figax=(fig, ax), show=False)
#
#    predict = rnn.predict(X_seed)
#    for emb in predict:
#        print(word_emb.find_closest(emb,1))
#
#    rnn = RNN(
#        hidden_activation='Tanh()',
#        output_activation='Identity()',
#        loss_function='mse()',
#        optimiser='SGD()',
#        regression=True,
#        threshold=1,
#        )
#
#    whole_sequence_output, hidden_state = rnn.fit(
#        X, y, epo,
#        num_hidden_nodes=hidden_nodes, return_sequences=True,
#        independent_samples=True, learning_rate=learning_rate_curr
#        )
#
#    rnn.plot_loss(plt, figax=(fig, ax), show=False)
#
#    predict = rnn.predict(X_seed)
#    for emb in predict:
#        print(word_emb.find_closest(emb,1))
#
#    rnn = RNN(
#        hidden_activation='Tanh()',
#        output_activation='Identity()',
#        loss_function='mse()',
#        optimiser='SGD_momentum()',
#        regression=True,
#        threshold=1,
#        )
#
#    whole_sequence_output, hidden_state = rnn.fit(
#        X, y, epo,
#        num_hidden_nodes=hidden_nodes, return_sequences=True,
#        independent_samples=True, learning_rate=learning_rate_curr,
#        momentum_rate=0.9)
#
#    rnn.plot_loss(plt, figax=(fig, ax), show=False)
#
#    predict = rnn.predict(X_seed)
#    for emb in predict:
#        print(word_emb.find_closest(emb,1))
#
#    rnn = RNN(
#        hidden_activation='Tanh()',
#        output_activation='Identity()',
#        loss_function='mse()',
#        optimiser='RMSProp()',
#        regression=True,
#        threshold=1,
#        )
#
#    whole_sequence_output, hidden_state = rnn.fit(
#        X, y, epo,
#        num_hidden_nodes=hidden_nodes, return_sequences=True,
#        independent_samples=True, learning_rate=learning_rate_curr,
#        decay_rate=0.001)
#
#    rnn.plot_loss(plt, figax=(fig, ax), show=True)
#
#    predict = rnn.predict(X_seed)
#    for emb in predict:
#        print(word_emb.find_closest(emb,1))

# plt.plot(rnn.get_stats()['loss'])
# plt.show()
# x_seed = X[0][0]
# print(word_emb.find_closest(x_seed, number=1))
# ret = rnn.predict(x_seed, hidden_state, 10)

# for emb in ret:
#     word = word_emb.find_closest(emb, number=1)
#     print(word)
