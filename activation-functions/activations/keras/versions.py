"""
This file contains the mathematical implementations of the rational activation function versions
a,b,c and d.
"""
import tensorflow as tf


def _get_xps(in_tensor, numerator_weights, denominator_weights):
    """
    creates a list of ascending powers of x

    :param in_tensor: input tensor
    :param numerator_weights: vector containing the weights a_0, ... a_n
    :param denominator_weights: vector containing the weights b_0, ... b_m
    :return: a list that looks approximately like this [1-tensor, x, x^2, ... x^m]
    """
    # create a list
    xps = list()
    # append the input tensor to the list
    xps.append(in_tensor)
    # add x^2, x^3, ... x^{max(n,m)} to the list
    for _ in range(max(numerator_weights.shape[0], denominator_weights.shape[0])):
        xps.append(xps[-1] * in_tensor)
    # inserts a tensor that is shaped like x, but contains only 1s as the first element
    xps.insert(0, tf.ones_like(in_tensor))
    return xps


def _version_a(in_tensor, numerator_weights, denominator_weights, training):
    """
    version a of rational activation function

    f(x) = p(x) / q(x) = (a_0 + a_1 * x + a_2 * x^2 + ... + a_n * x^n) /
                (1 + |b_0 * x| + | b_1 * x^2| + ... + | b_m * x^{m+1}|)

    note: q(x) contains m absolute value terms here

    :param in_tensor: input tensor
    :param numerator_weights: vector containing the weights a_0, ... a_n
    :param denominator_weights: vector containing the weights b_0, ... b_m
    :param training: (NOT IN USE) whether the call is in inference mode or training mode
    :return: f(x), i.e. the input tensor with the rational activation function applied to it
    """

    xps = _get_xps(in_tensor, numerator_weights, denominator_weights)

    numerator = 0
    for i in range(numerator_weights.shape[0]):
        w_n = numerator_weights[i]
        numerator = numerator + w_n * xps[i]

    denominator = 1.0
    for j in range(denominator_weights.shape[0]):
        w_d = denominator_weights[j]
        denominator = denominator + tf.abs(w_d * xps[j + 1])

    return numerator / denominator


def _version_b(in_tensor, numerator_weights, denominator_weights, training):
    """
    version b of rational activation function

    f(x) = p(x) / q(x) = (a_0 + a_1 * x + a_2 * x^2 + ... + a_n * x^n) /
                (1 + |b_0 * x + b_1 * x^2 + ... + b_m * x^{m + 1}|)

    note: q(x) contains only one absolute value term here

    :param in_tensor: input tensor
    :param numerator_weights: vector containing the weights a_0, ... a_n
    :param denominator_weights: vector containing the weights b_0, ... b_m
    :param training: (NOT IN USE) whether the call is in inference mode or training mode
    :return: f(x), i.e. the input tensor with the rational activation function applied to it
    """

    xps = _get_xps(in_tensor, numerator_weights, denominator_weights)

    numerator = 0
    for i in range(numerator_weights.shape[0]):
        w_n = numerator_weights[i]
        numerator = numerator + w_n * xps[i]

    denominator = 0
    for j in range(denominator_weights.shape[0]):
        w_d = denominator_weights[j]
        denominator = denominator + w_d * xps[j + 1]

    return numerator / (1 + tf.abs(denominator))


def _version_c(in_tensor, numerator_weights, denominator_weights, training):
    """
    version c of rational activation function

    f(x) = p(x) / q(x) = (a_0 + a_1 * x + a_2 * x^2 + ... + a_n * x^n) /
                (0.1 + |b_0 + b_1 * x + b_2 * x^2 + ... + b_m * x^m|)

    note: q(x) contains a variable term (epsilon) here, and also a b_0 term

    :param in_tensor: input tensor
    :param numerator_weights: vector containing the weights a_0, ... a_n
    :param denominator_weights: vector containing the weights b_0, ... b_m
    :param training: (NOT IN USE) whether the call is in inference mode or training mode
    :return: f(x), i.e. the input tensor with the rational activation function applied to it
    """

    xps = _get_xps(in_tensor, numerator_weights, denominator_weights)

    numerator = 0
    for i in range(numerator_weights.shape[0]):
        w_n = numerator_weights[i]
        numerator = numerator + w_n * xps[i]

    denominator = 0
    for j in range(denominator_weights.shape[0]):
        w_d = denominator_weights[j]
        denominator = denominator + w_d * xps[j]

    return numerator / (0.1 + tf.abs(denominator))


def _version_d(in_tensor, numerator_weights, denominator_weights, training, random_deviation=0.1):
    """
    version d of rational activation function

    f(x) = p(x) / q(x) =
    (noised(a_0) + noised(a_1) * x + noised(a_2) * x^2 + ... + noised(a_n) * x^n) /
                (1 + |noised(b_0) * x + noised(b_1) * x^2 + ... + noised(b_m) * X^{m+1}|)

    Noised parameters have uniform noise to be in range
    [(1-random_deviation)*parameter,(1+random_deviation)*parameter].

    :param in_tensor: input tensor
    :param numerator_weights: vector containing the weights a_0, ... a_n
    :param denominator_weights: vector containing the weights b_0, ... b_m
    :param training: (NOT IN USE) whether the call is in inference mode or training mode
    :param random_deviation: random deviation
    :return: f(x), i.e. the input tensor with the rational activation function applied to it

    """
    # if in training mode, apply Function B
    if not training:
        return _version_b(in_tensor, numerator_weights, denominator_weights, training)

    # else: inference mode
    # get list of polynomial [1, X, X^2, X^3....X^n]
    xps = _get_xps(in_tensor, numerator_weights, denominator_weights)

    # replace None wiht 1 in in_tensor.shape to avoid value error
    input_shape = [1 if x is None else x for x in in_tensor.shape]

    # assign weights to coefficients of numerator of polynomial
    numerator = 0
    for i in range(numerator_weights.shape[0]):
        # assign noise factor with uniform distribution
        noise = tf.random.uniform(shape=input_shape, minval=1 - random_deviation,
                                  maxval=1+random_deviation, dtype=tf.dtypes.float32)
        w_n_noised = numerator_weights[i] * noise
        numerator = numerator + w_n_noised * xps[i]

    # assign weights to coefficients of denominator of polynomial
    denominator = 0
    for j in range(denominator_weights.shape[0]):
        # assign noise factor with uniform distribution
        noise = tf.random.uniform(shape=input_shape, minval=1 - random_deviation,
                                  maxval=1+random_deviation, dtype=tf.dtypes.float32)
        w_d_noised = denominator_weights[j] * noise
        denominator = denominator + w_d_noised * xps[j + 1]

    return numerator / (1 + tf.abs(denominator))
