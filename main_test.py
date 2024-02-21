import sys
import git
import numpy as np
from rnn.rnn import ReccurentNN
from utils.activations import Relu, Softmax, Tanh
path_to_root = git.Repo('.', search_parent_directories=True).working_dir
sys.path.append(path_to_root)

rnn = ReccurentNN(Relu(), Softmax())
# y_true = np.array([[[1, 2, 3, 4, 5]], [[2, 3, 4, 5, 6]], [3, 4, 5, 6, 7]])
# x = np.array([[[0, 1, 2, 3, 4]], [[1, 2, 3, 4, 5]], [[2, 3, 4, 5, 6]]], dtype=float)
# y_true = np.array([[[1, 2], [1, 3]], [[1, 2], [1, 3]]])
# x = np.array([[[1, 2], [1, 3]], [[1, 2], [1, 3]]], dtype=float)
# y_true = np.array([[[1], [1]], [[1], [1]]])
# x = np.array([[[1], [1]], [[1], [1]]], dtype=float)

# y_true = np.array([[[1, 1, 2, 2]],
#                    [[1, 1, 2, 2]],
#                    [[1, 1, 2, 2]],
#                    [[1, 1, 2, 2]],
#                    [[1, 1, 2, 2]],
#                    [[1, 1, 2, 2]],
#                    [[1, 1, 2, 2]],
#                    [[1, 1, 2, 2]],
#                    [[1, 1, 2, 2]],
#                    [[1, 1, 2, 2]],
#                    [[1, 1, 2, 2]],
#                    [[1, 1, 2, 2]],
#                    [[1, 1, 2, 2]],
#                    [[1, 1, 2, 2]],
#                    [[1, 1, 2, 2]],
#                    [[1, 1, 2, 2]],
#                    [[1, 1, 2, 2]],
#                    [[1, 1, 2, 2]],
#                    [[1, 1, 2, 2]],
#                    [[1, 1, 2, 2]],
#                    [[1, 1, 2, 2]],
#                    [[1, 1, 2, 2]],
#                    [[1, 1, 2, 2]],
#                    [[1, 1, 2, 2]],
#                    [[1, 1, 2, 2]],
#                    [[1, 1, 2, 2]],
#                    [[1, 1, 2, 2]],
#                    [[1, 1, 2, 2]],
#                    [[1, 1, 2, 2]],
#                    [[1, 1, 2, 2]],
#                    [[1, 1, 2, 2]],
#                    [[1, 1, 2, 2]],
#                    [[1, 1, 2, 2]],
#                    [[1, 1, 2, 2]],
#                    [[1, 1, 2, 2]],
#                    [[1, 1, 2, 2]],
#                    [[1, 1, 2, 2]],
#                    [[1, 1, 2, 2]],
#                    [[1, 1, 2, 2]],
#                    [[1, 1, 2, 2]],
#                    [[1, 1, 2, 2]],
#                    [[1, 1, 2, 2]],
#                    [[1, 1, 2, 2]],
#                    [[1, 1, 2, 2]],
#                    [[1, 1, 2, 2]],
#                    [[1, 1, 2, 2]],
#                    [[1, 1, 2, 2]],
#                    [[1, 1, 2, 2]],
#                    [[1, 1, 2, 2]],
#                    [[1, 1, 2, 2]],
#                    [[1, 1, 2, 2]],
#                    [[1, 1, 2, 2]],
#                    [[1, 1, 2, 2]],
#                    [[1, 1, 2, 2]],
#                    [[1, 1, 2, 2]],
#                    [[1, 1, 2, 2]],
#                    [[1, 1, 2, 2]],
#                    [[1, 1, 2, 2]],
#                    [[1, 1, 2, 2]],
#                    [[1, 1, 2, 2]],
#                    [[2, 2, 3, 3]]], dtype=int)

# x = np.array([[[0, 1, 2, 3]],
#               [[0, 1, 2, 3]],
#               [[0, 1, 2, 3]],
#               [[0, 1, 2, 3]],
#               [[0, 1, 2, 3]],
#               [[0, 1, 2, 3]],
#               [[0, 1, 2, 3]],
#               [[0, 1, 2, 3]],
#               [[0, 1, 2, 3]],
#               [[0, 1, 2, 3]],
#               [[0, 1, 2, 3]],
#               [[0, 1, 2, 3]],
#               [[0, 1, 2, 3]],
#               [[0, 1, 2, 3]],
#               [[0, 1, 2, 3]],
#               [[0, 1, 2, 3]],
#               [[0, 1, 2, 3]],
#               [[0, 1, 2, 3]],
#               [[0, 1, 2, 3]],
#               [[0, 1, 2, 3]],
#               [[0, 1, 2, 3]],
#               [[0, 1, 2, 3]],
#               [[0, 1, 2, 3]],
#               [[0, 1, 2, 3]],
#               [[0, 1, 2, 3]],
#               [[0, 1, 2, 3]],
#               [[0, 1, 2, 3]],
#               [[0, 1, 2, 3]],
#               [[0, 1, 2, 3]],
#               [[0, 1, 2, 3]],
#               [[0, 1, 2, 3]],
#               [[0, 1, 2, 3]],
#               [[0, 1, 2, 3]],
#               [[0, 1, 2, 3]],
#               [[0, 1, 2, 3]],
#               [[0, 1, 2, 3]],
#               [[0, 1, 2, 3]],
#               [[0, 1, 2, 3]],
#               [[0, 1, 2, 3]],
#               [[0, 1, 2, 3]],
#               [[0, 1, 2, 3]],
#               [[0, 1, 2, 3]],
#               [[0, 1, 2, 3]],
#               [[0, 1, 2, 3]],
#               [[0, 1, 2, 3]],
#               [[0, 1, 2, 3]],
#               [[0, 1, 2, 3]],
#               [[0, 1, 2, 3]],
#               [[0, 1, 2, 3]],
#               [[0, 1, 2, 3]],
#               [[0, 1, 2, 3]],
#               [[0, 1, 2, 3]],
#               [[0, 1, 2, 3]],
#               [[0, 1, 2, 3]],
#               [[0, 1, 2, 3]],
#               [[0, 1, 2, 3]],
#               [[0, 1, 2, 3]],
#               [[0, 1, 2, 3]],
#               [[0, 1, 2, 3]],
#               [[0, 1, 2, 3]],
#               [[1, 2, 3, 4]]], dtype=float)
# rnn.fit(x, y_true, 100, 0.1)


def file_to_single_string(file_path):
    with open(file_path, 'r') as file:
        # Read lines, remove newlines and any empty or whitespace-only lines
        lines = [line.strip() for line in file if line.strip()]
        # Concatenate all lines into a single string
        single_string = ' '.join(lines)
    return single_string


# Example usage
file_path = path_to_root + '/data/cat_text'
text_string = file_to_single_string(file_path)

data = text_string
chars = list(set(data))

char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for i, ch in enumerate(chars)}
print(ix_to_char)
n = 0
p = 0
n_chars = len(chars)

y_true = []
X_train = []
while n < 100:
    if p + n_chars + 1 >= len(data) or n == 0:
        p = 0
    inputs = [char_to_ix[ch] for ch in data[p:p+n_chars]]
    targets = [char_to_ix[ch] for ch in data[p+1:p+n_chars+1]]
    y_true.append(np.array(targets))
    X_train.append(np.array(inputs))
    p += n_chars  # move data pointer
    n += 1  # iteration counter

X_train = np.array(X_train)

y_true = np.array(y_true)

rnn.fit(X_train, y_true, 100, 0.1)

ret = rnn.predict(X_train, 1, 100)
txt = ''.join(ix_to_char[ix] for ix in ret)
print('----\n %s \n----' % (txt, ))
