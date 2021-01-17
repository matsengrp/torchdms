.. _sec_tutorials:

============
Tutorials
============

.. todo::
  Provide some information on the required data format of ``torchdms`` and perhaps a tutorial on how to go from a CSV file to a pickled object.

.. todo::
  Prepping data for model fitting in ``torchdms.data``.

.. todo::
  Creating models with ``torchdms.model``.

.. todo::
  Training models with ``torchdms.analysis`` and ``torchdms.loss``.

.. todo::
  Evaluating model performance with ``torchdms.evaluation`` and ``torchdms.plot``.

++++++++++++++
CLI tutorials
++++++++++++++

The command line interface is called ``tdms``, and has nested subcommands.

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
