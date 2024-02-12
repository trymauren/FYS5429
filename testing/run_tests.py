import git
import sys
from testing import tests
path_to_root = git.Repo('.', search_parent_directories=True).working_dir
sys.path.append(path_to_root)


tests.test_activations()
tests.test_read_save()
