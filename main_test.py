import sys
import git
import numpy as np
from rnn.rnn import ReccurentNN
from utils.activations import Relu

import hydra
from omegaconf import DictConfig

path_to_root = git.Repo('.', search_parent_directories=True).working_dir
sys.path.append(path_to_root)

# rnn = ReccurentNN(Relu())
# y_true = np.array([[1, 3, 2, 4, 5], [3, 2, 4, 1, 5]])
# x = np.array([[[0, 1, 3, 2, 4]], [[0, 3, 2, 4, 1]]], dtype=float)
# # y_true = np.ones((2, 26))
# # x = np.ones((2, 3, 1))
# rnn.fit(x, y_true, 100, 0.1)

# # print(np.array([1,1]).reshape(-1,1).shape)

# le = np.array([[2, 2], [1, 1]])
# print(le[0])

@hydra.main(version_base = None, config_path = "conf", config_name = "config")
def main_test(cfg : DictConfig) -> None:
	x = np.array([[[0, 1, 3, 2, 4]], [[0, 3, 2, 4, 1]]], dtype=float)
	y_true = np.array([[1, 3, 2, 4, 5], [3, 2, 4, 1, 5]])
	rnn = ReccurentNN(cfg)
	rnn.fit(x, y_true)

if __name__ == "__main__":
	main_test()
