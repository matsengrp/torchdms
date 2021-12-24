.. _sec_tutorials:

============
Tutorials
============


***************************************************
Preparing DMS data for analysis with ``torchdms``
***************************************************


To start analyzing deep mutational scanning data with ``torchdms``, ensure you have a data frame with the following columns:

* **aa_substitutions**: contains a list of space separated strings of the amino acid substitutions observed
* **n_aa_substitutions**: an integer indicating the number of amino acid substitutions observed
* **barcode**: a string that indicates the nucleotide barcode used

An example of what an appropriate data frame should look like is found below:


.. csv-table::
   :header: "library", "barcode", "aa_substitutions", "n_aa_substitutions", "func_score"
   :widths: 10, 20, 20, 10, 10

   "lib1", "ACGT", "N1L I2P", 2, -2.1
   "lib1", "TCAA", " ", 0, 0.0
   "lib1", "TCGA", "T3A", 1, 0.7


Once your data is in this format, you will then need to compress this data frame as well as the wild-type sequence into a pickle file:
An example of how to do this with a pandas data frame named ``df`` and a string of the wildtype sequence ``wtseq`` can be found below:


.. code-block:: python

  import pickle

  with open("data/test_df.pkl", "wb") as f:
    pickle.dump([df, wtseq], f)


Once this pickle file has been prepared, you can begin ``torchdms`` analysis with either the command line interface (CLI) or python API, instructions for both can be found below.


**************
CLI tutorials
**************

The command line interface is called ``tdms``, and has nested subcommands.
Below we walk through an example of using ``tdms`` to conduct an analysis consisting of the following steps:

1. Preparing a DMS dataset for analysis
2. Creating a model
3. Training a model
4. Creating plots of fitting results

This analysis will be conducted on a simulated dummy dataset used for developer testing, located in ``torchdms/torchdms/data/test_df.pkl``.


.. _sec_tdms_prep:

Preparing a dataset for analysis with ``tdms prep``
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

This step in the protocol involves partitioning a DMS dataset into training, validation, and testing sets.
``torchdms`` partitions DMS data by *strata*, where we define a *stratum* as the set of all variants that have the same number of amino acid substitutions.
More details on the partitioning process can be found in the documentation for ``torchdms.data``.

There are three required arguments for the ``tdms prep`` command that must be entered in order:

1. *IN_PATH* - the path to the pickle file to be partitioned
2. *OUT_PREFIX* - a path for the prepped data to be saved
3. *TARGETS* - the column name(s) in the pickled data frame that we want to predict

Running the command below in the install directory creates a partitioned dataset at ``torchdms/data/test_df.prepped.pkl`` such that:

* The testing and validation datasets will have 10 unique variants from each stratum and the remaining variants will be placed in the training dataset.
* Any stratum with less than 30 unique variants will not be included in the analysis


.. code-block:: console

   tdms prep torchdms/data/test_df.pkl torchdms/data/test_df.prepped affinity_score \
   --per-stratum-variants-for-test 10 \
   --skip-stratum-if-count-is-smaller-than 30


.. _sec_tdms_create:

Creating models with ``tdms create``
++++++++++++++++++++++++++++++++++++

This step involves defining the model architecture to be used in the rest of the analysis.
Details on the different model classes and hyper-parameters can be found in the documentation for ``torchdms.model``.

There are three required arguments for the ``tdms create`` command that must be entered in order:

1. *DATA_PATH* - the path to the pickle file to be partitioned
2. *OUT_PATH* - a path for the model object to be saved
3. *MODEL_STRING* - a string describing the model architecture to be used

Running the command below will create a ``FullyConnected`` model with 1 latent node and a nonlinear transformation consisting of 1 hidden layer with 10 relu-activated nodes:

.. code-block:: console

   tdms create torchdms/data/test_df.prepped.pkl run.model "FullyConnected;[1,10];[None,'relu']"


.. _sec_tdms_train:

Training models with ``tdms train``
++++++++++++++++++++++++++++++++++++

Now we will train the model on the partitioned data we created above.

There are two required arguments for the ``tdms train`` command that must be entered in order:

1. *MODEL_PATH* - the path to the saved model
2. *DATA_PATH* - the path to the partitioned dataset

Running the following command will train the model and save it to the original location along with a pickle file of details concerning the training.

.. code-block:: console

    tdms train run.model torchdms/data/test_df.prepped.pkl


.. _sec_tdms_scatter:

Creating plots to asses model performance on unseen variants with ``tdms scatter``
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


This command uses a ``torchdms`` model and makes fitness predictions on a testing dataset, creating a scatter plot of the predictions against the observed fitness scores.

There are two required arguments for the ``tdms scatter`` command that must be entered in order:

1. *MODEL_PATH* - the path to the saved model
2. *DATA_PATH* - the path to the partitioned dataset

There is also a required option for writing the output to a file:

1. *out* - a prefix for the scatterplot and correlations for each stratum to be saved

Running the following command will use the model to create a scatterplot of out-of-sample fitness predictions vs the observed fitness scores, and save it to the *scatter.png* and *scatter.corr.csv*.

.. code-block:: console

    tdms scatter run.model torchdms/data/test_df.prepped.pkl --out scatter


.. _sec_tdms_beta:

Creating a heatmap of inferred mutational effects with ``tdms beta``
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

This command plots a heatmap of the latent mutational effects inferred by the model, which we refer to as *beta coefficients*.

There are two required arguments for the ``tdms beta`` command that must be entered in order:

1. *MODEL_PATH* - the path to the saved model
2. *DATA_PATH* - the path to the partitioned dataset

There is also a required option for writing the output to a file:

1. *out* - a prefix for the heatmap to be saved

Running the following command will plot the model's beta coefficients in a file called *beta.png*.

.. code-block:: console

    tdms beta run.model torchdms/data/test_df.prepped.pkl --out beta


.. _sec_tdms_heatmap:

Creating a heatmap of single-mutant predictions with ``tdms heatmap``
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

This command plots a heatmap of the fitness predictions for each single variant by the model, which includes the beta coefficient as well as any nonlinear transformation.


There are two required arguments for the ``tdms heatmap`` command that must be entered in order:

1. *MODEL_PATH* - the path to the saved model
2. *DATA_PATH* - the path to the partitioned dataset

There is also a required option for writing the output to a file:

1. *out* - a prefix for the heatmap to be saved

Running the following command will plot the model's single-mutant fitness predictions in a file called *smps.png*.

.. code-block:: console

    tdms heatmap run.model torchdms/data/test_df.prepped.pkl --out smps


.. _sec_tdms_geplot:

Plotting the learned nonlinearity of a model with ``tdms geplot``
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

This command plots the shape of the nonlinearity learned by the model.

There are two required arguments for the ``tdms geplot`` command that must be entered in order:

1. *MODEL_PATH* - the path to the saved model
2. *DATA_PATH* - the path to the partitioned dataset

There is also a required option for writing the output to a file:

1. *out* - a prefix for the global epistasis plot to be saved

Running the following command will plot the model's additive latent space against the model predictions and observed testing variants in a file called *geplot.png*.

.. code-block:: console

    tdms geplot run.model torchdms/data/test_df.prepped.pkl --out geplot

.. note::
  This command only works with models that have no more than 2 latent layer nodes, and some nonlinear transformation.


.. _sec_tdms_go:

Running a full analysis with ``tdms go``
+++++++++++++++++++++++++++++++++++++++++

You can run a complete ``tdms`` anaylsis with the ``tdms go`` command.
This command will run all of the commands above (except for ``tdms prep``), as well as some other model diagnostics.
To run ``tdms go``, you will need to specify a configuration file for the analysis in a JSON file.
For example, we could have the following contents in ``config.json``:

::

  {
      "default": {
          "data_path": "/test_df.prepped.pkl",
          "model_string": "FullyConnected;[1, 10];['sigmoid', 'relu']",
          "prefix": "_ignore/run",
          "beta_l1_coefficients": "1",
          "epochs": 10,
          "seed": 42
      }
  }

The above JSON file will do the following in the analysis:

* Use the prepped dataset at ``data_path``
* Create a model architecture defined by ``model_string``
* Dump all output files to a directory *_ignore/*, all with a prefix of *run.*
* Apply an L1 penalty to the model's beta coefficients during training, with a λ = 1
* Train the model for 10 epochs
* Use a random seed of 42 throughout the analysis

To run the analysis, run:

.. code-block:: console

    tdms go --config config.json


.. note::
  To see all CLI options and arguments, please reference the CLI documentation.

**************
API tutorials
**************

.. note::
  Maybe move the API tutorial to a separate page that holds a notebook?

.. todo::
  Prepping data for model fitting in ``torchdms.data``.

.. todo::
  Creating models with ``torchdms.model``.

.. todo::
  Training models with ``torchdms.analysis`` and ``torchdms.loss``.

.. todo::
  Evaluating model performance with ``torchdms.evaluation`` and ``torchdms.plot``.
