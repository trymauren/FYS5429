from importlib.resources import open_text
from pathlib import Path
import sys
from typing import Dict
import git
import numpy as np
from collections.abc import Callable
import yaml
from utils.activations import Relu, Tanh
from utils.loss_functions import Mean_Square_Loss as mse
from utils.optimisers import Adam
from utils import read_load_model
from utils import construct_yaml
import matplotlib.pyplot as plt
from rnn.rnn import RNN
import click
from importlib import import_module

path_to_root = git.Repo('.', search_parent_directories=True).working_dir
sys.path.append(path_to_root)

class rnn_not_nonsense():

    def __init__(self,train : bool = True, config: Dict | Path | str = "default"):
        self.config = self._load_config(config)
        print(type(self.config))
        self.rnn = RNN(**self.config['init'])
        
        if train:
            self.rnn.fit(**self.config['fit'])


    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.rnn.predict(X)
    

    def fit(self,
            X: np.ndarray = None,
            y: np.ndarray = None,
            epochs: int = None,
            num_hidden_states: int = 5,
            num_hidden_nodes: int = 5,
            ) -> None:
        self.rnn.fit(X,y,epochs, num_hidden_states, num_hidden_nodes)


    @staticmethod
    def _load_config(config: Dict | Path | str = "default"):
        if config == 'default':
            with open(f'{path_to_root}/config_default.yml') as f:
                config = yaml.safe_load(f)
        if isinstance(config, dict):
            pass
        elif isinstance(config, (str, Path)):
            with open(config, 'r') as f:
                #yaml.load(f, Loader=construct_yaml.get_loader())
                config = yaml.safe_load(f)
        else:
            raise TypeError(
                'modules_config must either be set to "default", or be a dict,'
                ' or a path to a yaml file')
        return config

if __name__ == "__main__":
    rnn_not_nonsense()