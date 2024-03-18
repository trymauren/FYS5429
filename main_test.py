import sys
import git
import numpy as np
from rnn.rnn import ReccurentNN
from utils.activations import Relu, Tanh

path_to_root = git.Repo('.', search_parent_directories=True).working_dir
sys.path.append(path_to_root)

rnn = ReccurentNN(Tanh(), Tanh(), regression=True)

# x = np.array([np.sin(np.linspace(0, 2 * np.pi, 100))]).T
# y = np.array([np.sin(np.linspace(0, 2 * np.pi, 100))]).T
# ret = rnn.fit(x, y, 1000, 10)

x = np.ones((200, 5))
y = np.zeros((200, 5))
ret = rnn.fit(x, y, 100, num_hidden_states=10, num_hidden_layers=20)

print(ret)
