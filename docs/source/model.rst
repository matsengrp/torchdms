##################
Specifying models
##################

``torchdms`` offers multiple network architecture templates to specify models.
When calling ``tdms create`` or specifying a model in a JSON configuration file, you must specify a model string with your desired architecture.
Different model classes have different model strings, below are a few examples of how to define various types of models.

********************************
Models for any output dimension
********************************

The following examples define models capable of predicting output of an arbitrary dimension.

FullyConnected
---------------

To build a purely fully connected deep neural network with two layers (i.e., a 20 unit sigmoid activated layer followed by a 10 unit ReLU activated layer):

::

    "FullyConnected(20,sigmoid,10,relu)"

To build a typical global epistasis model with a one dimensional, additive latent space and a nonlinearity of 10 sigmoids:

::

    "FullyConnected(1,identity,10,sigmoid)"

If you wanted to add a second layer to the nonlinearity, say 15 ReLU units:

::

    "FullyConnected(1,identity,10,sigmoid,15,relu)"

To use a two-dimensional latent space in the previous model:

::

    "FullyConnected(2,identity,10,sigmoid,15,relu)"

Nonlinear interaction modules can also be added by specifying a sub-architecture before the latent space.
These modules run parallel from the latent space, and their outputs are concatenated to the latent representation of a sequence and passed to the nonlinearity.

To add a nonlinear interaction module of 20 ReLU units followed by 10 more ReLU units to the previous architecture:

::

    "FullyConnected(20,relu,10,relu,1,identity,10,sigmoid)"


**********************************
Models for two-dimensional output
**********************************

The following examples define models specialized to the case of two outputs.
We have developed these with the idea of the first output being binding and the second being stability.

Independent
------------

The ``Independent`` model replicates two ``FullyConnected`` models, each with a model string like that given to the ``FullyConnected`` model.
So, to build model with parallel and independent tracks of a one-dimensional latent space and 10 sigmoid nonlinearity for each output:

::

    "Independent(1,identity,10,sigmoid)"

Conditional
------------

Defining ``Conditional`` models without an interaction module is similar to defining an ``Independent`` model:

::

    "Conditional(1,identity,10,sigmoid)"

Likewise, adding an interaction module follows the same syntax:

::

    "Conditional(20,relu,10,relu,1,identity,10,sigmoid)"

While the model string of ``Conditional`` and ``Independent`` are the same, it is important to note that in the ``Conditional`` architecture, the stability component of the latent space also influences the binding nonlinearity.
We'll add something here eventually, but for now see `this issue comment for a diagram <https://github.com/matsengrp/torchdms/pull/75#issuecomment-672309225>`_ (in that comment it's called ``Sparse2D`` but we thought that ``Conditional`` was a better name).

ConditionalSequential
----------------------

To train the sub-networks of a ``Conditional`` model sequentially (stability then binding):

::

    "ConditionalSequential(1,identity,10,sigmoid)"

And with nonlinear interaction modules:

::

    "ConditionalSequential(20,relu,10,relu,1,identity,10,sigmoid)"
