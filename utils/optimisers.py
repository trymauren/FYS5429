import sys
import git
import numpy as np
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

def clip_gradient(gradient_vector: np.ndarray, threshold: float) -> np.ndarray:
	"""
	Finds l2-norm of gradient vector and normalizes it.
	TODO Find out if actual delta parameters are the one to be adjusted 
	to make norm of grad vector be within threshold, or if just scaling the 
	grad vector itself suffices
	"""
	grad_norm = np.linalg.norm(gradient_vector)
	#Only need positive threshold check as l2 norm ensues we only get 
	#positive norm values
	if grad_norm > threshold:
		gradient_vector = (threshold/grad_norm)*gradient_vector
	else:
		return gradient_vector
	return gradient_vector