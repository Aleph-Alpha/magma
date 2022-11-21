"""
this file tests that cuda and cpu results are consistent
"""
import mxnet as mx
from mxnet.context import gpu, cpu, num_gpus
from mxnet.ndarray import LeakyReLU, tanh, sigmoid
from numpy import all, isclose

from rational.mxnet import Rational


def _test_consistency(version: str, approx_func, sym: bool):
    """
    test rational activation function from keras package on test_data,
    validating that cuda and cpu results are consistent, i.e. that there is no significant
    difference between cuda and cpu results

    :param sym: use symbolic execution if True, else imperative execution
    :param approx_func: which function to use as initial shape
    :param version: which version of the function to test
    """
    # declare results
    cpu_result = None
    gpu_result = None

    # set cpu context and test
    with mx.Context(cpu(0)):
        # instantiate a tensor for testing
        test_data = mx.nd.array([-2., -1, 0., 1., 2.])

        init_fun_names = {LeakyReLU: 'leaky_relu', tanh: 'tanh', sigmoid: 'sigmoid'}

        # instantiate rational activation function under test on cpu
        cpu_fut = Rational(approx_func=init_fun_names.get(approx_func), version=version,
                           cuda=False, trainable=False)

        # create small neural networks and add futs as layers
        cpu_net = mx.gluon.nn.HybridSequential()
        with cpu_net.name_scope():
            cpu_net.add(cpu_fut)
        cpu_net.initialize()

        # trigger symbolic rather than imperative API, if specified
        if sym:
            cpu_net.hybridize()

        # run the function on test data
        cpu_result = cpu_net(test_data)

    # set gpu context and test
    assert num_gpus() > 0, 'tried to run on GPU, but none available.'
    with mx.Context(gpu(0)):
        # instantiate a tensor for testing
        test_data = mx.nd.array([-2., -1, 0., 1., 2.])

        init_fun_names = {LeakyReLU: 'leaky_relu', tanh: 'tanh', sigmoid: 'sigmoid'}

        # instantiate rational activation function under test on gpu
        gpu_fut = Rational(approx_func=init_fun_names.get(approx_func), version=version,
                           cuda=True, trainable=False)

        # create small neural networks and add futs as layers
        gpu_net = mx.gluon.nn.HybridSequential()
        with gpu_net.name_scope():
            gpu_net.add(gpu_fut)
        gpu_net.initialize()

        # trigger symbolic rather than imperative API, if specified
        if sym:
            gpu_net.hybridize()

        # run the function on test data
        gpu_result = gpu_net(test_data)

    # check that there is no significant difference between the results
    assert all(isclose(cpu_result.asnumpy(), gpu_result.asnumpy(), atol=1e-06))


def test_a_for_consistency_lrelu_nd():
    _test_consistency(version='A', approx_func=LeakyReLU, sym=False)


def test_a_for_consistency_tanh_nd():
    _test_consistency(version='A', approx_func=tanh, sym=False)


def test_a_for_consistency_sigmoid_nd():
    _test_consistency(version='A', approx_func=sigmoid, sym=False)


def test_b_for_consistency_lrelu_nd():
    _test_consistency(version='B', approx_func=LeakyReLU, sym=False)


def test_b_for_consistency_tanh_nd():
    _test_consistency(version='B', approx_func=tanh, sym=False)


def test_b_for_consistency_sigmoid_nd():
    _test_consistency(version='B', approx_func=sigmoid, sym=False)


def test_c_for_consistency_lrelu_nd():
    _test_consistency(version='C', approx_func=LeakyReLU, sym=False)


def test_c_for_consistency_tanh_nd():
    _test_consistency(version='C', approx_func=tanh, sym=False)


def test_c_for_consistency_sigmoid_nd():
    _test_consistency(version='C', approx_func=sigmoid, sym=False)


def test_d_for_consistency_lrelu_nd():
    _test_consistency(version='D', approx_func=LeakyReLU, sym=False)


def test_d_for_consistency_tanh_nd():
    _test_consistency(version='D', approx_func=tanh, sym=False)


def test_d_for_consistency_sigmoid_nd():
    _test_consistency(version='D', approx_func=sigmoid, sym=False)


def test_a_for_consistency_lrelu_sym():
    _test_consistency(version='A', approx_func=LeakyReLU, sym=True)


def test_a_for_consistency_tanh_sym():
    _test_consistency(version='A', approx_func=tanh, sym=True)


def test_a_for_consistency_sigmoid_sym():
    _test_consistency(version='A', approx_func=sigmoid, sym=True)


def test_b_for_consistency_lrelu_sym():
    _test_consistency(version='B', approx_func=LeakyReLU, sym=True)


def test_b_for_consistency_tanh_sym():
    _test_consistency(version='B', approx_func=tanh, sym=True)


def test_b_for_consistency_sigmoid_sym():
    _test_consistency(version='B', approx_func=sigmoid, sym=True)


def test_c_for_consistency_lrelu_sym():
    _test_consistency(version='C', approx_func=LeakyReLU, sym=True)


def test_c_for_consistency_tanh_sym():
    _test_consistency(version='C', approx_func=tanh, sym=True)


def test_c_for_consistency_sigmoid_sym():
    _test_consistency(version='C', approx_func=sigmoid, sym=True)


def test_d_for_consistency_lrelu_sym():
    _test_consistency(version='D', approx_func=LeakyReLU, sym=True)


def test_d_for_consistency_tanh_sym():
    _test_consistency(version='D', approx_func=tanh, sym=True)


def test_d_for_consistency_sigmoid_sym():
    _test_consistency(version='D', approx_func=sigmoid, sym=True)
