"""
Rational Activation Functions for MXNET
=======================================

This module allows you to create Rational Neural Networks using Learnable
Rational activation functions with MXNET networks.
"""
import mxnet as mx
from mxnet import initializer
from mxnet.gluon import HybridBlock

from rational.utils.get_weights import get_parameters
from rational.mxnet.versions import _version_a, _version_b, _version_c, _version_d
from rational._base.rational_base import Rational_base


class Rational(Rational_base, HybridBlock):
    """
    Rational Activation Function, inheriting from ``mxnet.gluon.HybridBlock``.

    Arguments:
            approx_func (str):
                The name of the approximated function for initialisation. \n
                The different functions are available in `rational.rationals_config.json`. \n
                Default: ``leaky_relu``

            degrees (tuple of int):
                The degrees of the numerator (P) and denominator (Q).\n
                Default ``(5, 4)``

            cuda (bool):
                whether to execute on cuda device.\n
                NOTE: THIS PARAMETER IS CURRENTLY NOT CONSIDERED.\n
                CUDA GPUS ARE USED WHEN IT IS POSSIBLE

            version (str):
                Version of Rational to use. Rational(x) = P(x)/Q(x),
                where
                P(x) = (a_0 + a_1 * x + a_2 * x^2 + ... + a_n * x^n) and \n

                `A`: Q(x) = (1 + \|b_0 * x\| + \| b_1 * x^2\| + ... + \| b_m * x^{m+1}\|)\n
                `B`: Q(x) = (1 + \|b_0 * x + b_1 * x^2 + ... + b_m * x^{m + 1}\|)\n
                `C`: Q(x) = (0.1 + \|b_0 + b_1 * x + b_2 * x^2 + ... + b_m * x^m\|)\n
                `D`: like `B` with noised coefficients b_i\n

                Default ``A``

            trainable (bool):
                Whether the weights are trainable, i.e, if they are updated during
                backward pass. \n
                Default ``True``

    Returns:
        HybridBlock:
            Rational hybrid block
    """

    def __init__(self, approx_func='leaky_relu', degrees=(5, 4), cuda=False,
                 version='A', trainable=True, name=None, **kwargs):
        
        if name is None:
            name = approx_func
            super().__init__(name)
        # super(Rational, self).__init__(**kwargs)

        # read initial parameter configuration from external files
        w_numerator, w_denominator = get_parameters(
            version, degrees, approx_func)

        # convert w_numerator and w_denominator to mxnet arrays
        w_numerator = mx.nd.array(w_numerator)
        w_denominator = mx.nd.array(w_denominator)

        # register the amount of weights in numerator and denominator, since we need them during
        # symbolic execution, but are unable to retrieve them at later stages
        self.numerator_length = len(w_numerator)
        self.denominator_length = len(w_denominator)
        self.training = trainable
        self.degrees = degrees
        self.version = version
        self.init_approximation = approx_func

        # set specified context (currently not happening, since unclear, how and why helpful)
        # self.device = gpu() if cuda else cpu()

        # register and configure weights (numerator and denominator coefficients)
        with self.name_scope():
            self.numerator = self.params.get(name='w_numerator', shape=(len(w_numerator),),
                                             init=initializer.Constant(
                                                 w_numerator),
                                             grad_req='write' if trainable
                                             else 'null',
                                             differentiable=trainable)
            self.denominator = self.params.get(name='w_denominator', shape=(len(w_denominator),),
                                               init=initializer.Constant(
                                                   w_denominator),
                                               grad_req='write' if trainable
                                               else 'null',
                                               differentiable=trainable)

        # register whether function is trainable, since this information needs to be passed to
        # version D
        self.training = trainable

        self.init_approximation = approx_func

        # set rational activation function version
        self.rational_func = {'A': _version_a, 'B': _version_b, 'C': _version_c, 'D': _version_d} \
            .get(version)
        if self.rational_func is None:
            raise ValueError(
                "rational activation function version %s not implemented" % version)

    def hybrid_forward(self, F, x, numerator, denominator):
        return self.rational_func(F, x, numerator, denominator, self.training,
                                  self.numerator_length, self.denominator_length)

    def numpy(self):
        """
        Returns a numpy version of this activation function.
        """
        from rational.numpy import Rational as Rational_numpy
        rational_n = Rational_numpy(self.init_approximation, self.degrees,
                                    self.version)
        rational_n.numerator = self.numerator.data().asnumpy().tolist()
        rational_n.denominator = self.denominator.data().asnumpy().tolist()
        return rational_n

    @property
    def device(self):
        return str(mx.context.current_context())
