import sys
import git
path_to_root = git.Repo('.', search_parent_directories=True).working_dir
sys.path.append(path_to_root)


class Optimiser():
	pass

class SGD(Optimiser):
	pass

class Adam(Optimiser):
	pass
 
class lbfgs(Optimiser): 
	pass