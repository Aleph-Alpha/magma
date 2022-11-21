"""
This file consists test for the backward pass of rational activation functions.
I.e. it is tested here whether they are in fact trainable
"""
import tensorflow as tf
import numpy as np

from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from rational.keras import Rational

# load MNIST data set
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Converts a class vector (integers) to binary class matrix.
y_cat_test = to_categorical(y=y_test, num_classes=10)
y_cat_train = to_categorical(y=y_train, num_classes=10)

# normalize
x_train = x_train / x_train.max()  # /255
x_test = x_test / x_test.max()

# increase color channel
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)


def _test_backward_template(version: str, function_is_layer: bool):
    """
    This method serves as a base method for all backward tests.
    It tests for a given version of Rational whether Rational can be integrated into a small
    tensorflow model. It also tests whether the rational activation function's coefficients
    (weights) are updated

    :param version: version of the rational activation function
    :param function_is_layer: declare function as a tf.keras.layer in the model if True.
    Else declare as the activation parameter for a given tf.keras.layer
    """
    # create our rational activation function under test
    fut = Rational(version=version)

    # check that the coefficients of the Rational module are in fact trainable
    assert fut.numerator.trainable
    assert fut.denominator.trainable

    # retrieve the initial coefficient values
    nums_before_training = fut.numerator.read_value()
    dens_before_training = fut.denominator.read_value()

    # create the model
    if function_is_layer:
        model = tf.keras.Sequential([
            # convolutional layer
            tf.keras.layers.Conv2D(filters=32, kernel_size=(
                4, 4), input_shape=(28, 28, 1)),

            # rational activation function as layer
            fut,

            # transform 2D to 1D with flatten
            tf.keras.layers.Flatten(),

            # output layer
            tf.keras.layers.Dense(10, activation='softmax')
        ])
    else:
        model = tf.keras.Sequential([
            # convolutional layer
            tf.keras.layers.Conv2D(filters=32, kernel_size=(
                4, 4), input_shape=(28, 28, 1), activation=fut),

            # transform 2D to 1D with flatten
            tf.keras.layers.Flatten(),

            # output layer
            tf.keras.layers.Dense(10, activation='softmax')
        ])

    # compile the model
    model.compile(loss='categorical_crossentropy')

    # train with the first ten samples
    model.fit(x_train[:10], y_cat_train[:10])

    # check that the coefficient values changed
    nums_after_training = fut.numerator.read_value()
    dens_after_training = fut.denominator.read_value()

    # check that at least one coefficient changed in numerators
    assert not np.all(tf.equal(nums_before_training, nums_after_training))
    # check that at least one coefficient changed in denominators
    assert not np.all(tf.equal(dens_before_training, dens_after_training))


def test_a_backward_as_layer():
    _test_backward_template('A', True)


def test_b_backward_as_layer():
    _test_backward_template('B', True)


def test_c_backward_as_layer():
    _test_backward_template('C', True)


def test_d_backward_as_layer():
    _test_backward_template('D', True)


def test_a_backward_as_param():
    _test_backward_template('A', False)


def test_b_backward_as_param():
    _test_backward_template('B', False)


def test_c_backward_as_param():
    _test_backward_template('C', False)


def test_d_backward_as_param():
    _test_backward_template('D', False)
