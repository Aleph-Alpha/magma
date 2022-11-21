"""
This file contains methods that are useful for multiple test files in this directory
"""
import numpy as np
import tensorflow as tf
from tensorflow.nn import leaky_relu

from rational.keras import Rational


def _activation(func, data):
    """
    apply activation function to data

    :param func: activation function
    :param data: data to be applied
    """
    if func == leaky_relu:
        return np.array(func(data, alpha=0.01))
    return np.array(func(data))


def _test_template(version: str, approx_func, cuda: bool):
    """
    compare the result of Rational activation function with expected result

    :param cuda: whether to execute on cuda
    :param version: which version of Rational activation function to test
    """
    # configure the device
    device_type = 'GPU' if cuda else 'CPU'
    logical_devices = tf.config.list_logical_devices(device_type)

    if len(logical_devices) == 0:
        raise Exception('{} device not available for testing. \n Available devices: {}.'
                        .format(device_type, str(tf.config.list_logical_devices())))

    # execute the remaining test on the specified device
    with tf.device(logical_devices[0].name):
        # instantiate tensor for testing purpose
        test_data = [-2., -1, 0., 1., 2.]
        test_tensor = tf.convert_to_tensor(np.array(test_data, np.float32), np.float32)

        # instantiate expected results of activation function
        expected_res = _activation(approx_func, test_data)

        # instantiate Rational activation function with specific version
        trainable = False  # if version != 'D' else True
        rational = Rational(approx_func=approx_func.__name__, version=version,
                            cuda=cuda, trainable=trainable)(test_tensor).numpy()

        assert np.all(np.isclose(rational, expected_res, atol=5e-02))
