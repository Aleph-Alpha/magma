================================================
Welcome to Rational Activations's documentation!
================================================

.. highlight:: python

Rational Activations provided in this package are learnable rational activation
function to create rational neural networks.
So far, only the pytorch and MXNET implementation are available.

Requirements:
#############
This project depends on:

- pytorch
- numpy, scipy (if you want to add different initially approximated functions)
- matplotlib (if you want to use the plotting properties)
- torchvision (if you want to convert a torchvision model architecture to Rational)

Download and install:
You can download from the
`Github <https://github.com/ml-research/activation_functions>`_ repository or:

::

    pip3 install rational-activations

To use it:

.. literalinclude:: tutorials/code/how_to_use_rationals.py
   :lines: 1-6

.. toctree::
    :maxdepth: 2
    :caption: Tutorials:
    :glob:

    tutorials/*


.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: API:
   :glob:

   rational_modules/*

.. toctree::
   :maxdepth: 2
   :caption: Contents:

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


Cite us in your paper
=====================

Rational activations were first introduced as Padé Activation Units in
:cite:p:`molina2019pade`, and later developed as Rational Activation in
:cite:p:`delfosse2021recurrent`.

To cite the package only: :cite:p:`delfosse2020rationals`.

.. bibliography::
   :all:
