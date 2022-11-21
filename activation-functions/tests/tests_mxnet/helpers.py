"""
This file contains methods that are useful for multiple test files in this directory
"""
import mxnet as mx
from mxnet.context import gpu, cpu, num_gpus
from mxnet.ndarray import LeakyReLU, tanh, sigmoid
from numpy import all, isclose

from rational.mxnet import Rational


def _activation(func, data):
    """
    apply activation function to data

    :param func: activation function
    :param data: data to be applied
    """
    if func == LeakyReLU:
        return func(data, slope=0.01)
    return func(data)


def _test_template(version: str, approx_func, cuda: bool, sym: bool):
    """
    compare the result of Rational activation function with expected result

    :param sym: use symbolic execution if True, else imperative execution
    :param cuda: whether to execute on cuda
    :param version: which version of Rational activation function to test
    """
    # set context to either GPU or CPU
    if cuda:
        assert num_gpus() > 0, 'tried to run on GPU, but none available.'

    device = gpu(0) if cuda else cpu(0)
    with mx.Context(device):
        # instantiate tensor for testing purpose
        test_data = mx.nd.array([-2., -1, 0., 1., 2.])

        init_fun_names = {LeakyReLU: 'leaky_relu', tanh: 'tanh', sigmoid: 'sigmoid'}
        # instantiate rational activation function under test (fut) with given version,
        # initial approximation, and type of execution in a small neural network
        fut = Rational(approx_func=init_fun_names.get(approx_func), version=version,
                       cuda=cuda, trainable=False)

        # create small neural network and add fut as layer
        net = mx.gluon.nn.HybridSequential()
        with net.name_scope():
            net.add(fut)
        net.initialize()

        # trigger symbolic rather than imperative API, if specified
        if sym:
            net.hybridize()

        # run fut on test data
        res = net(test_data)

        # compute expected results for comparison
        expected_res = _activation(approx_func, test_data)

        assert all(isclose(res.asnumpy(), expected_res.asnumpy(), atol=5e-02))
