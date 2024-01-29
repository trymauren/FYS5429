import numpy as np
import git
import importlib
import sys
path_to_root = git.Repo('.', search_parent_directories=True).working_dir
sys.path.append(path_to_root)


def test_activations():
    from utils import activations as act

    relu = act.Relu()
    sigmoid = act.Sigmoid()
    tanh = act.Tanh()

    z = np.ones((5,2))
    assert(relu(z).shape==z.shape)
    assert(tanh(z).shape==z.shape)
    assert(sigmoid(z).shape==z.shape)

    z = np.array([-100,-2,-1,0,1,2,100])
    assert(relu(z).all() == np.array([0,0,0,0,1,2,100]).all())
    assert(tanh(z).all() == np.array([-1.0,-0.9640275800758169,-0.7615941559557649,
                                0.0,0.7615941559557649,0.9640275800758169,1.0]).all())

    np.testing.assert_allclose(sigmoid(z) == np.array([0.0,0.11920292202211755,
                                                             0.26894142136999512,
                                                             0.5,
                                                             0.73105857863000487,
                                                             0.880797077977882444,
                                                             0.999999999999999]), 1e-10,1e10)

def test_read_save():
    from utils import read_load_model as rsm
    d = dict()
    d[2] = 'test'
    obj = [2,3,1,42,d]
    rsm.save_model(obj, path_to_root + '/saved_models', 'test_model')
    assert(obj == rsm.load_model(path_to_root + '/saved_models' + '/test_model'))


