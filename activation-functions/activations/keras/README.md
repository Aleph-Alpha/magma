# Rational Activation Functions for Tensorflow & Keras
This package contains an implementation of [Rational Activation Functions](https://arxiv.org/abs/1907.06732)
for the machine learning framework [TensorFlow](https://www.tensorflow.org), based on the [Keras API](https://www.tensorflow.org/api_docs/python/tf/keras), which is fully integrated
into TensorFlow.

## Integrating Rational Activation Functions into Neural Networks
In TensorFlow, you can instantiate a Rational Activation Function by running
```python
from rational.keras import Rational

my_fun = Rational()
```

This instantiates a `Layer`.

## Customizing `Rational`
If you wish to customize your `Rational` instance, feel free to play around with its parameters.
```python
from rational.keras import Rational

my_costum_fun = Rational(approx_func='tanh')
```

## Integrating a `Rational` instance into a neural network

There are two ways to integrate a `Rational` instance into a neural network.

### Option 1
You can pass your `Rational` instance as the `activation` parameter of a layer. In contrast to most other activation
functions, it will of course be learnable.
```python
import tensorflow as tf

from rational.keras import Rational

my_fun = Rational()

model = tf.keras.Sequential([
    # layers ...

    # rational activation function as a parameter of some other layer (here: Dense)
    tf.keras.layers.Dense(100, activation=my_fun)
    
    # more layers ...
])
```
### Option 2
You can add your `Rational` instance as an additional learnable layer to a network.
```python
import tensorflow as tf

from rational.keras import Rational

my_fun = Rational()

model = tf.keras.Sequential([
    # layers ...

    # rational activation function as layer
    my_fun
    
    # more layers ...
])
```


## Documentation
Please find more documentation on [ReadTheDocs](https://rational-activations.readthedocs.io).
