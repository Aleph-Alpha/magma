"""
This file contains the mathematical implementations of the rational activation function versions
a,b,c and d.
"""
from mxnet import current_context


def _get_xps_num(F, x, weights_len):
    """
    creates a sequence of ascending powers of x for numerator

    :param weights_len: int, amount of weights, needed for symbolic execution
    :param F: a function space either mxnet.nd or mxnet.sym
    :param x: input sequence of scalars
    :return: a two-dimensional sequence that looks approximately like this
     [[1,1,...], [--x--], [--x^2--],... , [--x^{weights_len}--]], where x is a vector (sequence
     of scalars)
    """
    #  create an array containing ones
    xps = F.expand_dims(F.ones_like(x), axis=0)

    # append arrays containing x, x^2, ... x^n to the list
    for i in range(weights_len - 1):
        factor = F.sum(F.ones(shape=(1, i + 1)))
        x_i = F.expand_dims(F.broadcast_power(x, factor), axis=0)
        xps = F.concat(xps, x_i, dim=0)

    return xps


def _get_xps_denom(F, x, weights_len):
    """
    creates a sequence of ascending powers of x for denominator

    :param weights_len: int, amount of weights, needed for symbolic execution
    :param F: a function space either mxnet.nd or mxnet.sym
    :param x: input sequence of scalars
    :return: a two-dimensional sequence that looks approximately like this
     [[--x--], [--x^2--],... , [--x^n--], [--x^{weights_len + 1}--]], where x is a vector (sequence
     of scalars)
    """
    #  create an array containing x
    xps = F.expand_dims(x, axis=0)

    # append arrays containing x^2, ... x^{n+1} to the list
    for i in range(weights_len - 1):
        factor = F.sum(F.ones(shape=(1, i + 2)))
        x_i = F.expand_dims(F.broadcast_power(x, factor), axis=0)
        xps = F.concat(xps, x_i, dim=0)

    return xps


def _compute_p(F, x, num_len, numerator_weights):
    # get powers of x for numerator weights, flatten (relevant if x is multidimensional)
    xps_num = F.flatten(_get_xps_num(F, x, num_len))
    # expand dimension of numerator_weights
    numerator_weights = F.expand_dims(numerator_weights, axis=1)
    # multiply numerator_weights with the powers of x
    prod = F.broadcast_mul(xps_num, numerator_weights)
    # compute the sum over the product
    return F.sum(prod, axis=0)


def _version_a(F, x, numerator_weights, denominator_weights, training, num_len, denom_len):
    """
    version a of rational activation function

    f(x) = p(x) / q(x) = (a_0 + a_1 * x + a_2 * x^2 + ... + a_n * x^n) /
                (1 + |b_0 * x| + | b_1 * x^2| + ... + | b_m * x^{m+1}|)

    note: q(x) contains m absolute value terms here

    :param num_len: int, amount of numerator weights. Needed for symbolic execution
    :param denom_len: int, amount of denominator weights. Needed for symbolic execution
    :param F: a function space either mxnet.nd or mxnet.sym
    :param x: input sequence of scalars
    :param numerator_weights: vector containing the weights a_0, ... a_n
    :param denominator_weights: vector containing the weights b_0, ... b_m
    :param training: (NOT IN USE) whether the call is in inference mode or training mode
    :return: f(x), i.e. the input tensor with the rational activation function applied to it
    """
    # COMPUTE P
    p = _compute_p(F, x, num_len, numerator_weights)

    # COMPUTE Q
    # flatten xps (relevant if multidimensional)
    xps_den = F.flatten(_get_xps_denom(F, x, denom_len))
    # expand dimension of denominator_weights
    denominator_weights = F.expand_dims(denominator_weights, axis=1)
    # multiply denominator_weights with the powers of x
    prod = F.broadcast_mul(xps_den, denominator_weights)
    # compute the absolute value
    abs_prod = F.abs(prod)
    # compute the sum
    sum_abs_prod = F.sum(abs_prod, axis=0)
    # add one to each element
    ones = F.ones_like(sum_abs_prod)

    q = F.elemwise_add(ones, sum_abs_prod)

    # compute p / q
    result_flat = F.elemwise_div(p, q)
    # reshape to original shape of x (relevant if multidimensional)
    return F.reshape_like(result_flat, x)


def _version_b(F, x, numerator_weights, denominator_weights, training, num_len, denom_len):
    """
    version b of rational activation function

    f(x) = p(x) / q(x) = (a_0 + a_1 * x + a_2 * x^2 + ... + a_n * x^n) /
                (1 + |b_0 * x + b_1 * x^2 + ... + b_m * x^{m + 1}|)

    note: q(x) contains only one absolute value term here

    :param num_len: int, amount of numerator weights. Needed for symbolic execution
    :param denom_len: int, amount of denominator weights. Needed for symbolic execution
    :param F: a function space either mxnet.nd or mxnet.sym
    :param x: input sequence of scalars
    :param numerator_weights: vector containing the weights a_0, ... a_n
    :param denominator_weights: vector containing the weights b_0, ... b_m
    :param training: (NOT IN USE) whether the call is in inference mode or training mode
    :return: f(x), i.e. the input tensor with the rational activation function applied to it
    """
    # COMPUTE P
    p = _compute_p(F, x, num_len, numerator_weights)

    # COMPUTE Q
    # get powers of x for denominator weights, flatten (relevant if x is multidimensional)
    xps_den = F.flatten(_get_xps_denom(F, x, denom_len))
    # expand dimension of denominator_weights
    denominator_weights = F.expand_dims(denominator_weights, axis=1)
    # multiply denominator_weights with powers of x
    prod = F.broadcast_mul(xps_den, denominator_weights)
    # compute the sum
    sum_prod = F.sum(prod, axis=0)
    # compute the absolute value
    abs_sum_prod = F.abs(sum_prod)
    # add one to each element
    ones = F.ones_like(abs_sum_prod)

    q = F.elemwise_add(ones, abs_sum_prod)

    # compute p / q
    result_flat = F.elemwise_div(p, q)
    # reshape to original shape of x (relevant if multidimensional)
    return F.reshape_like(result_flat, x)


def _version_c(F, x, numerator_weights, denominator_weights, training, num_len, denom_len):
    """
    version c of rational activation function

    f(x) = p(x) / q(x) = (a_0 + a_1 * x + a_2 * x^2 + ... + a_n * x^n) /
                (0.1 + |b_0 + b_1 * x + b_2 * x^2 + ... + b_m * x^m|)

    note: q(x) contains a variable term (epsilon) here, and also a b_0 term

    :param num_len: int, amount of numerator weights. Needed for symbolic execution
    :param denom_len: int, amount of denominator weights. Needed for symbolic execution
    :param F: a function space either mxnet.nd or mxnet.sym
    :param x: input sequence of scalars
    :param numerator_weights: vector containing the weights a_0, ... a_n
    :param denominator_weights: vector containing the weights b_0, ... b_m
    :param training: (NOT IN USE) whether the call is in inference mode or training mode
    :return: f(x), i.e. the input tensor with the rational activation function applied to it
    """
    # COMPUTE P
    p = _compute_p(F, x, num_len, numerator_weights)

    # COMPUTE Q
    # get powers of x for denominator weights, flatten (relevant if x is multidimensional)
    xps_den = F.flatten(_get_xps_num(F, x, denom_len))
    # expand dimensions of denominator_weights
    denominator_weights = F.expand_dims(denominator_weights, axis=1)
    # multiply denominator_weights with powers of x
    prod = F.broadcast_mul(xps_den, denominator_weights)
    # compute the sum
    sum_prod = F.sum(prod, axis=0)
    # compute the absolute value
    abs_sum_prod = F.abs(sum_prod)

    # add epsilon (here 0.1) to each element
    ones = F.ones_like(abs_sum_prod)
    factor = F.sum(F.ones(shape=(1, 10)))
    epsilons = F.broadcast_div(ones, factor)

    q = F.elemwise_add(epsilons, abs_sum_prod)

    # compute p / q
    result_flat = F.elemwise_div(p, q)
    # reshape to original shape of x (relevant if multidimensional)
    return F.reshape_like(result_flat, x)


def _version_d(F, x, numerator_weights, denominator_weights, training, num_len, denom_len,
               random_deviation=0.1):
    """
    version d of rational activation function

    f(x) = p(x) / q(x) =
    (noised(a_0) + noised(a_1) * x + noised(a_2) * x^2 + ... + noised(a_n) * x^n) /
                (1 + |noised(b_0) * x + noised(b_1) * x^2 + ... + noised(b_m) * X^{m+1}|)

    Noised parameters have uniform noise to be in range
    [(1-random_deviation)*parameter,(1+random_deviation)*parameter].

    :param num_len: int, amount of numerator weights. Needed for symbolic execution
    :param denom_len: int, amount of denominator weights. Needed for symbolic execution
    :param F: a function space either mxnet.nd or mxnet.sym
    :param x: input sequence of scalars
    :param numerator_weights: vector containing the weights a_0, ... a_n
    :param denominator_weights: vector containing the weights b_0, ... b_m
    :param training: (NOT IN USE) whether the call is in inference mode or training mode
    :param random_deviation: random deviation
    :return: f(x), i.e. the input tensor with the rational activation function applied to it

    """
    # if in training mode, apply "normal" version B, else apply noise to  weights
    if not training:
        # do not add noise
        return _version_b(F, x, numerator_weights, denominator_weights, False, num_len, denom_len)

    # COMPUTE P
    # apply noise to numerator weights
    noise = F.uniform(low=1 - random_deviation, high=1 + random_deviation, shape=num_len,
                      ctx=current_context())
    numerator_weights = F.elemwise_mul(numerator_weights, noise)

    p = _compute_p(F, x, num_len, numerator_weights)

    # COMPUTE Q
    # get powers of x for denominator weights, flatten (relevant if x is multidimensional)
    xps_den = F.flatten(_get_xps_denom(F, x, denom_len))

    # apply noise to denominator weights
    noise = F.uniform(low=1 - random_deviation, high=1 + random_deviation, shape=denom_len,
                      ctx=current_context())
    denominator_weights = F.elemwise_mul(denominator_weights, noise)
    # expand dimension of denominator_weights
    denominator_weights = F.expand_dims(denominator_weights, axis=1)
    # multiply denominator_weights with powers of x
    prod = F.broadcast_mul(xps_den, denominator_weights)
    # compute the sum
    sum_prod = F.sum(prod, axis=0)
    # compute the absolute value
    abs_sum_prod = F.abs(sum_prod)
    # add one to each element
    ones = F.ones_like(abs_sum_prod)

    q = F.elemwise_add(ones, abs_sum_prod)

    # compute p / q
    result_flat = F.elemwise_div(p, q)
    # reshape to original shape of x (relevant if multidimensional)
    return F.reshape_like(result_flat, x)
