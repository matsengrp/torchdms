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
