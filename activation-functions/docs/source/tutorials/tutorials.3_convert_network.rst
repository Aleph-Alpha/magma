Apply the functions to torchvision model
========================================

In this tutorial, we are going to apply rational functions to a (pretrained) PyTorch network directly taken from the torchvision library.

First, we import the torchvision library:

.. literalinclude:: code/convert_network.py
   :lines: 1

Then let's instantiate a torchvision model on the first cuda-enabled GPU:

.. literalinclude:: code/convert_network.py
   :lines: 3-4

Now we can take the VGG-16 model and convert all its ReLU layers to Rationals with the help of a utility function:

.. literalinclude:: code/convert_network.py
   :lines: 6-7

The function will scan the network for certain activation functions and will replace them with rationals that are initialized to approximate the same function.
