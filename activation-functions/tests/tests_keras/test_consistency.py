"""
this file tests that cuda and cpu results are consistent
"""
import tensorflow as tf
import numpy as np

from tensorflow.nn import leaky_relu
from tensorflow.math import tanh, sigmoid
from rational.keras import Rational

# instantiate a tensor for testing (from numpy array)
test_tensor = tf.convert_to_tensor(
    np.array([-2., -1, 0., 1., 2.], np.float32), np.float32)


def _test_consistency(version: str, approx_func):
    """
    test rational activation function from keras package on test_tensor,
    validating that cuda and cpu results are consistent, i.e. that there is no significant
    difference between cuda and cpu results

    :param approx_func: which function to use as initial shape
    :param version: which version of the function to test
    """

    # instantiate rational activation functions under test on cpu and cuda
    trainable = False
    cpu_fut = Rational(version=version, cuda=False, approx_func=approx_func.__name__) \
        if version != 'D' else Rational(version=version, cuda=False,
                                        approx_func=approx_func.__name__, trainable=trainable)

    cuda_fut = Rational(version=version, cuda=True, approx_func=approx_func.__name__) \
        if version != 'D' else Rational(version=version, cuda=True,
                                        approx_func=approx_func.__name__, trainable=trainable)
    # run the functions under test on our test tensor
    cpu_result = cpu_fut(test_tensor).numpy()
    cuda_result = cuda_fut(test_tensor).numpy()

    # check that there is no significant difference between the results
    assert np.all(np.isclose(cpu_result, cuda_result, atol=1e-06))


def test_a_for_consistency_lrelu():
    _test_consistency(version='A', approx_func=leaky_relu)


def test_a_for_consistency_tanh():
    _test_consistency(version='A', approx_func=tanh)


def test_a_for_consistency_sigmoid():
    _test_consistency(version='A', approx_func=sigmoid)


def test_b_for_consistency_lrelu():
    _test_consistency(version='B', approx_func=leaky_relu)


def test_b_for_consistency_tanh():
    _test_consistency(version='B', approx_func=tanh)


def test_b_for_consistency_sigmoid():
    _test_consistency(version='B', approx_func=sigmoid)


def test_c_for_consistency_lrelu():
    _test_consistency(version='C', approx_func=leaky_relu)


def test_c_for_consistency_tanh():
    _test_consistency(version='C', approx_func=tanh)


def test_c_for_consistency_sigmoid():
    _test_consistency(version='C', approx_func=sigmoid)


def test_d_for_consistency_lrelu():
    _test_consistency(version='D', approx_func=leaky_relu)


def test_d_for_consistency_tanh():
    _test_consistency(version='D', approx_func=tanh)


def test_d_for_consistency_sigmoid():
    _test_consistency(version='D', approx_func=sigmoid)
