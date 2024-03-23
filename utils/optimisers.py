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
    TODO Find out if actual delta parameters are the ones to be adjusted 
    to make norm of grad vector be within threshold, or if just scaling 
    the grad vector itself suffices
    EDIT: found what seems to be an answer to exactly how the clipping 
    is done, it seems it's only scaling of the actual gradient: 
    https://towardsdatascience.com/what-is-gradient-clipping-b8e815cdfb48
    g = g*(threshold/l2norm(g)) or g = threshold*(g/l2norm(g))
    """
    grad_norm = np.linalg.norm(gradient_vector, ord=2, axis=0)
    #Only need positive threshold check as l2 norm ensues we only get 
    #positive norm values
    if grad_norm > threshold:
        gradient_vector = gradient_vector * (threshold/grad_norm)
    else:
        return gradient_vector
    return gradient_vector