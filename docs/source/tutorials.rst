.. _sec_tutorials:

==================================================
Preparing DMS data for analysis with ``torchdms``
==================================================


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


Once this pickle file has been prepared, you can begin ``torchdms`` analysis with either the command line interface (CLI) or python API.
