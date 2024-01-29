import git
import importlib
import sys
path_to_root = git.Repo('.', search_parent_directories=True).working_dir
sys.path.append(path_to_root)
from testing import tests


tests.test_activations()
tests.test_read_save()