"""
This file tests that cpu calculations produce correct results.
"""
from mxnet.ndarray import LeakyReLU, tanh, sigmoid

from .helpers import _activation, _test_template

# test cpu execution
CUDA = False


def test_a_on_cpu_lrelu_nd():
    _test_template(version='A', approx_func=LeakyReLU, cuda=CUDA, sym=False)


def test_b_on_cpu_lrelu_nd():
    _test_template(version='B', approx_func=LeakyReLU, cuda=CUDA, sym=False)


def test_c_on_cpu_lrelu_nd():
    _test_template(version='C', approx_func=LeakyReLU, cuda=CUDA, sym=False)


def test_d_on_cpu_lrelu_nd():
    _test_template(version='D', approx_func=LeakyReLU, cuda=CUDA, sym=False)


def test_a_on_cpu_tanh_nd():
    _test_template(version='A', approx_func=tanh, cuda=CUDA, sym=False)


def test_b_on_cpu_tanh_nd():
    _test_template(version='B', approx_func=tanh, cuda=CUDA, sym=False)


def test_c_on_cpu_tanh_nd():
    _test_template(version='C', approx_func=tanh, cuda=CUDA, sym=False)


def test_d_on_cpu_tanh_nd():
    _test_template(version='D', approx_func=tanh, cuda=CUDA, sym=False)


def test_a_on_cpu_sigmoid_nd():
    _test_template(version='A', approx_func=sigmoid, cuda=CUDA, sym=False)


def test_b_on_cpu_sigmoid_nd():
    _test_template(version='B', approx_func=sigmoid, cuda=CUDA, sym=False)


def test_c_on_cpu_sigmoid_nd():
    _test_template(version='C', approx_func=sigmoid, cuda=CUDA, sym=False)


def test_d_on_cpu_sigmoid_nd():
    _test_template(version='D', approx_func=sigmoid, cuda=CUDA, sym=False)


def test_a_on_cpu_lrelu_sym():
    _test_template(version='A', approx_func=LeakyReLU, cuda=CUDA, sym=True)


def test_b_on_cpu_lrelu_sym():
    _test_template(version='B', approx_func=LeakyReLU, cuda=CUDA, sym=True)


def test_c_on_cpu_lrelu_sym():
    _test_template(version='C', approx_func=LeakyReLU, cuda=CUDA, sym=True)


def test_d_on_cpu_lrelu_sym():
    _test_template(version='D', approx_func=LeakyReLU, cuda=CUDA, sym=True)


def test_a_on_cpu_tanh_sym():
    _test_template(version='A', approx_func=tanh, cuda=CUDA, sym=True)


def test_b_on_cpu_tanh_sym():
    _test_template(version='B', approx_func=tanh, cuda=CUDA, sym=True)


def test_c_on_cpu_tanh_sym():
    _test_template(version='C', approx_func=tanh, cuda=CUDA, sym=True)


def test_d_on_cpu_tanh_sym():
    _test_template(version='D', approx_func=tanh, cuda=CUDA, sym=True)


def test_a_on_cpu_sigmoid_sym():
    _test_template(version='A', approx_func=sigmoid, cuda=CUDA, sym=True)


def test_b_on_cpu_sigmoid_sym():
    _test_template(version='B', approx_func=sigmoid, cuda=CUDA, sym=True)


def test_c_on_cpu_sigmoid_sym():
    _test_template(version='C', approx_func=sigmoid, cuda=CUDA, sym=True)


def test_d_on_cpu_sigmoid_sym():
    _test_template(version='D', approx_func=sigmoid, cuda=CUDA, sym=True)
