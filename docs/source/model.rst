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

.. code-block:: python

    "FullyConnected;[20,10]['sigmoid','relu']"

To build a typical global epistasis model with a one dimensional, additive latent space and a nonlinearity of 10 sigmoids:

.. code-block:: python

    "FullyConnected;[1,10];[None,'sigmoid']"

If you wanted to add a second layer to the nonlinearity, say 15 ReLU units:

.. code-block:: python

    "FullyConnected;[1,10,15];[None,'sigmoid','relu']"

To use a two-dimensional latent space in the previous model:

.. code-block:: python

    "FullyConnected;[2,10,15];[None,'sigmoid','relu']"

Nonlinear interaction modules can also be added by specifying a sub-architecture before the latent space.
These modules run parallel from the latent space, and their outputs are concatenated to the latent representation of a sequence and passed to the nonlinearity.

To add a nonlinear interaction module of 20 ReLU units followed by 10 more ReLU units to the previous architecture:

.. code-block:: python

    "FullyConnected;[20,10,1,10];['relu','relu',None,'sigmoid']"


**********************************
Models for two-dimensional output
**********************************

The following examples define models specialized to the case of two outputs.
We have developed these with the idea of the first output being binding and the second being stability.

Independent
------------

The ``Independent`` model replicates two ``FullyConnected`` models, each with a model string like that given to the ``FullyConnected`` model.
So, to build model with parallel and independent tracks of a one-dimensional latent space and 10 sigmoid nonlinearity for each output:

.. code-block:: python

    "Independent;[1,10];[None,'sigmoid']"

Conditional
------------

Defining ``Conditional`` models without an interaction module is similar to defining an ``Independent`` model:

.. code-block:: python

    "Conditional;[1,10];[None,'sigmoid']"

Likewise, adding an interaction module follows the same syntax:

.. code-block:: python

    "Conditional;[20,10,1,10];['relu','relu',None,'sigmoid']"

While the model string of ``Conditional`` and ``Independent`` are the same, it is important to note that in the ``Conditional`` architecture, the stability component of the latent space also influences the binding nonlinearity.
We'll add something here eventually, but for now see `this issue comment for a diagram <https://github.com/matsengrp/torchdms/pull/75#issuecomment-672309225>`_ (in that comment it's called ``Sparse2D`` but we thought that ``Conditional`` was a better name).

ConditionalSequential
----------------------

To train the sub-networks of a ``Conditional`` model sequentially (stability then binding):

.. code-block:: python

    "ConditionalSequential;[1,10];[None,'sigmoid']"

And with nonlinear interaction modules:

.. code-block:: python

    "ConditionalSequential;[20,10,1,10];['relu','relu',None,'sigmoid']"


**************************************
Biophysical models for immune escape
**************************************


Escape
----------------------

To train a model for predicting antibody escape, the only argument is the numner of epitopes one wants to model. For example, a model with three epitopes:

.. code-block:: python

  "Escape;3"
