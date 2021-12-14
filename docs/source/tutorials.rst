.. _sec_tutorials:

============
Tutorials
============


***************************************************
Preparing DMS data for analysis with ``torchdms``
***************************************************

.. todo::
  Provide some information on the required data format of ``torchdms`` and perhaps a tutorial on how to go from a CSV file to a pickled object.

To start analyzing deep mutational scanning data with ``torchdms``, ensure you have a data frame with the following columns:

* **aa_substitutions**: contains a list of space separated strings of the amino acid substitutions observed
* **n_aa_substitutions**: an integer indicating the number of amino acid substitutions observed

Once you have a data frame with these required columns, feel free to name the modeling targets anything you like, for this example, we will be using `func_score` to represent the functional scores observed from the DMS.

You will then need to compress this data frame, along with a copy of the wild-type sequence into a pickle file.

.. todo::
  Link a notebook with a data-prep on simulated data.

An example of how to go from a raw csv file to the pickle file required for ``torchdms`` can be found here.

Once this pickle file has been prepared, you can begin ``torchdms`` analysis with either the command line interface (CLI) or python API, instructions for both can be found below.


**************
CLI tutorials
**************

The command line interface is called ``tdms``, and has nested subcommands.
Below is an example of using ``tdms`` to conduct an analysis consisting of the following steps:

1. Preparing a DMS dataset for analysis
2. Creating a model
3. Training a model
4. Creating plots of fitting results

.. todo::
  Prepping data for model fitting with ``tdms prep``.

.. _sec_tdms_prep:

Preparing a dataset for analysis with ``tdms prep``
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

This step in the protocol involves partitioning a DMS dataset into training, validation, and testing sets.
Details on the partitioning process can be found in the documentation for ``torchdms.data``.


.. _sec_tdms_create:

Creating models with ``tdms create``
++++++++++++++++++++++++++++++++++++

This step involves defining the model architecture to be used in the rest of the analysis.
Details on the different model classes and hyper-parameters can be found in the documentation for ``torchdms.model``

.. todo::
  Creating models with ``tdms create``.


.. _sec_tdms_train:

Training models with ``tdms train``
++++++++++++++++++++++++++++++++++++

.. todo::
  Training models with ``tdms train``


.. _sec_tdms_scatter:

Creating plots to asses model performance on unseen variants with ``tdms scatter``
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
.. todo::
  Evaluating model performance with ``tdms scatter``


.. _sec_tdms_beta:

Creating a heatmap of inferred mutational effects with ``tdms beta``
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


.. _sec_tdms_go:

Running a full analysis with ``tdms go``
+++++++++++++++++++++++++++++++++++++++++

.. todo::
  Running a complete analysis pipeline with ``tdms go``.



.. todo::
  The link below is broken or private. We also need a demo of how to create the pickle file—Zorian has a notebook we could use?

To use torchdms, first build a dataframe of mutations, and pickle that along with the wildtype sequence as for Tyler's `VRC01 data <https://github.com/jbloomlab/NIH45-46_DMS/blob/torchdms/affinity_expression_merge.ipynb>`_.

Then prepare data for use by torchdms (this example partitions by library and has both affinity and expression):

.. code-block:: console

    tdms prep --per-stratum-variants-for-test 500 \
        --skip-stratum-if-count-is-smaller-than 2000 \
        --partition-by library \
        df_and_wtseq.pkl \
        prepped \
        affinity_score expr_score

Now we create a model:

.. code-block:: console

    tdms create prepped_libA2.pkl my.model "Conditional;[100, 10, 1, 20];["relu", "relu", None, "relu"]"

We can train the model with various settings:

.. code-block:: console

    tdms train my.model prepped_libA2.pkl --independent-starts 0 --epochs 10

You now have a trained model which you can evaluate on test data using the ``error``, ``scatter``, ``geplot``, and ``beta`` subcommands.

Rather than use command line flags, you can use a JSON-format configuration file that might look like

::

    {
        "default": {
            "data_path": "/path/to/data.pkl",
            "loss_fn": "l1",
            "model_string": "Conditional;[100, 10, 1, 20];['relu', 'relu', None, 'relu']",
            "prefix": "run",
            "seed": 0
        }
    }

If you supply it to one of the subcommands using the ``--config`` flag, this will use the keys of the JSON as command line flags.
Note that hyphens in command line arguments become underscores in the JSON, for example ``loss-fn`` becomes ``loss_fn``.

You can build a model, train, and evaluate using ``tdms go``, which works well with such a JSON configuration file (say it is saved as ``config.json``):

.. code-block:: console

    tdms go --config config.json


**************
API tutorials
**************

.. todo::
  Prepping data for model fitting in ``torchdms.data``.

.. todo::
  Creating models with ``torchdms.model``.

.. todo::
  Training models with ``torchdms.analysis`` and ``torchdms.loss``.

.. todo::
  Evaluating model performance with ``torchdms.evaluation`` and ``torchdms.plot``.
