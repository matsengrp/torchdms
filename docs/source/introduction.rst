.. _sec_introduction:

============
Introduction
============

This is the documentation for ``torchdms``, a tool for inferring fitness landscapes of proteins from data obtained from Deep Mutational Scanning (DMS) experiments.
``torchdms`` allows for flexible modeling of DMS experiments with biophysically interpretable neural networks implemented in PyTorch.
This software implements commonly used models for DMS data, and is intended to facilitate and encourage the development of and novel models for learning the sequence-fitness relationships of proteins.

++++++++++++++++
Getting Started
++++++++++++++++

- To get started, follow the ``torchdms`` :ref:`Installation instructions <sec_installation>`.

- If you're interested in using the command line interface for ``torchdms``, check out the :ref:`toy example analysis here <sec_tutorials>`.

- If you're interested in using ``torchdms`` in your own python scripts or Jupyter notebooks, check out the :ref:`Tutorials page <sec_tutorials>`.


+++++++++++++
Background
+++++++++++++

Deep mutational scanning (DMS) is a powerful, high-throughput sequencing technique to obtain fitness measurements for hundreds of thousands of protein variants [#DMSreview]_, where `fitness is a quantitative measure of some functional capacity i.e.: expression levels, ligand binding, or antibody escape.
To enhance understanding of how sequence variation impacts protein fitness, several researchers have attempted to model the relationship between genotype and fitness (the fitness landscape) using DMS experiments.
Most attempts at inferring fitness landscapes fall into two broad categories: 1) simple, biophysically interpretable models and 2) expressive blackbox models.
Interpretable models of course provide scientists with the explanations for model predictions and allow for discovering new biology, these models are often too simple to accurately model the fitness landscape when multiple mutations are present.
On the other hand, the recent progress in the field of machine learning has allowed for more complicated and expressive models that are able to more accurately infer the fitness landscape of proteins further away from the original sequence, at the cost of model interpretability.

``torchdms`` attempts to find the middle ground between these two approaches, extending the global epistasis modeling framework [#GE]_.
To preserve model interpretability ``torchdms`` models fit a linear model to DMS data to infer latent mutational effects on fitness.
``torchdms`` then passes the output of this linear model into an arbitrary neural network to provide a more expressive model.
This modeling approach has been proven effective at inferring mutational effects from DMS data and has been instrumental in understanding the fitness effects of mutations on SARS-CoV-2 [#RBD_DMS]_ [#RBD_AbEscape]_ [#RBD_PolyEscape]_.
``torchdms`` is a flexible and modular modeling package that can be ran as its own analysis pipeline, or ported over to larger pipelines and projects.


+++++++++++++
References
+++++++++++++

.. [#DMSreview] Fowler, D., Fields, S. `Deep mutational scanning: a new style of protein science <https://doi.org/10.1038/nmeth.3027>`_. Nat Methods **11**, 801–807 (2014).

.. [#RBD_DMS] Starr, T. N. et al. `Deep mutational scanning of SARS-CoV-2 receptor binding domain reveals constraints on folding and ACE2 binding <https://doi.org/10.1016/j.cell.2020.08.012>`_. Cell **182**, 1295–1310.e20 (2020)

.. [#RBD_AbEscape] Greaney, A. J. et al. `Complete Mapping of Mutations to the SARS-CoV-2 Spike Receptor-Binding Domain that Escape Antibody Recognition <https://doi.org/10.1016/j.chom.2020.11.007>`_. Cell Host Microbe **29**, 44–57.e9 (2021)

.. [#RBD_PolyEscape] Greaney, A. J. et al. `Comprehensive mapping of mutations in the SARS-CoV-2 receptor-binding domain that affect recognition by polyclonal human plasma antibodies <https://doi.org/10.1016/j.chom.2021.02.003>`_. Cell Host Microbe **29**, 463–476.e6 (2021)

.. [#GE] Otwinowski, J., McCandlish, D. M. & Plotkin, J. B. `Inferring the shape of global epistasis <https://doi.org/10.1073/pnas.1804015115>`_. Proc. Natl. Acad. Sci. U. S. A. **115**, E7550–E7558 (2018)
