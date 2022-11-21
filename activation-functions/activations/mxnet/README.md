# Rational Activation Functions for MxNet
This package contains an implementation of [Rational Activation Functions](https://arxiv.org/abs/1907.06732)
for the machine learning framework [MxNet](https://mxnet.apache.org).

## Dependencies
Depending on your CUDA version, you should install ``mxnet-cu102`` (CUDA 10.2) or ``mxnet-cu101`` (CUDA 10.1). E.g.:

    pip3 install -U pip wheel
    pip3 install mxnet-cu102  rational-activations

## Integrating Rational Activation Functions into Neural Networks
In MxNet, you can instantiate a Rational Activation Function by running
```python
from rational.mxnet import Rational

my_fun = Rational()
```

This instantiates a `HybridBlock`, supporting both symbolic and imperative execution.

## Customizing `Rational`
If you wish to customize your `Rational` instance, feel free to play around with its parameters.
```python
from rational.mxnet import Rational

my_costum_fun = Rational(approx_func='tanh')
```

## Integrating a `Rational` instance into a neural network

You can integrate a `Rational` instance into a neural network as follows.
```python
import mxnet as mx
from rational.mxnet import Rational

my_fun = Rational()

# create small neural network and add my_fun as layer
net = mx.gluon.nn.HybridSequential()
with net.name_scope():
    net.add(my_fun)
net.initialize()

# for symbolic computation call 'hybridize'
net.hybridize()
```

## Documentation
Please find more documentation on [ReadTheDocs](https://rational-activations.readthedocs.io).
