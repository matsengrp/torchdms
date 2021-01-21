Specifying models
==================

``torchdms`` offers multiple network architecture templates to specify models.
When calling ``tdms create`` or specifying a model in a JSON configuration file, you must specify a model string with your desired architecture.
Different model classes have different model strings, below are a few examples of how to define various types of models.

FullyConnected
---------------

To build a typical global epistasis model with a one dimensional, additive latent space and a nonlinearity of 10 sigmoids:

.. code-block:: console

    "FullyConnected(1,identity,10,sigmoid)"

If you wanted to add a second layer to the nonlinearity, say 15 ReLU units:

.. code-block:: console

    "FullyConnected(1,identity,10,sigmoid,15,relu)"

To use a two-dimensional latent space in the previous model:

.. code-block:: console

    "FullyConnected(2,identity,10,sigmoid,15,relu)"


Independent
------------

Currently, the ``Independent`` model string is very similar to that of ``FullyConnected`` models, the provided architecture will be copied and used for a specific output dimension.
To build model with parallel and independent tracks of a one-dimensional latent space and 10 sigmoid nonlinearity for each output:

.. code-block:: console

    "Independent(1,identity,10,sigmoid)"

Nonlinear interaction modules can also be added by specifying a sub-architecture before the latent space.
These modules run parallel from the latent space, and their outputs are concatenated to the latent representation of a sequence and passed to the nonlinearity.

To add a nonlinear interaction module of 20 ReLU units followed by 10 more ReLU units to the previous architecture:

.. code-block:: console

    "Independent(20,relu,10,relu,1,identity,10,sigmoid)"

Conditional
------------

Defining ``Conditional`` models without an interaction module is similar to defining an ``Independent`` model:

.. code-block:: console

    "Conditional(1,identity,10,sigmoid)"

Likewise, adding an interaction module follows the same syntax:

.. code-block:: console

    "Conditional(20,relu,10,relu,1,identity,10,sigmoid)"

While the model string of ``Conditional`` and ``Independent`` are the same, it is important to note that in the ``Conditional`` architecture, the stability dedicated latent space also influences the binding nonlinearity.


ConditionalSequential
----------------------

To train the sub-networks of a ``Conditional`` model sequentially:

.. code-block:: console

    "ConditionalSequential(1,identity,10,sigmoid)"

And with nonlinear interaction modules:

.. code-block:: console

    "ConditionalSequential(20,relu,10,relu,1,identity,10,sigmoid)"
