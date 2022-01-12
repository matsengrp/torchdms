---
title: 'torchdms: A Python package for inferring genotype-phenotype maps'
tags:
- Python
- biology
- fitness
- deep mutational scanning
- deep learning
- PyTorch
- epistasis
- global epistasis
date: "4 January 2021"
output: pdf_document
authors:
- name: Zorian Thornton
  orcid: 0000-0002-2110-3119
  affiliation: 1, 2
- name: William S. DeWitt
  affiliation: 1, 2
- name: Jared Galloway
  affiliation: 1
- name: Jesse D. Bloom
  affiliation: 1, 2, 3
- name: Frederick A Matsen IV^[corresponding author]
  affiliation: 1, 2, 3
affiliations:
- name: Computational Biology Program, Fred Hutchinson Cancer Research Center
  index: 1
- name: Department of Genome Sciences, University of Washington
  index: 2
- name: Howard Hughes Medical Institute
  index: 2
bibliography: paper.bib
---


# Summary

The amino acid sequence that encodes a protein can have drastic effects on protein function.
When fitness is defined as the ability for a protein to perform a specific phenotype (i.e. binding to another protein), mutations made to the amino acid sequence of a protein will either: increase the fitness of the protein, decrease the fitness of the protein, or have little to no effect on the protein.
Understanding how amino acid changes impact proteins is essential for understanding protein variants that may cause disease, how and why pathogens are evolving to continue infecting hosts, and provide insight into some of the mechanisms underlying molecular evolution.

In this work, we introduce `torchdms`, a Python package for inferring fitness landscapes from data collected from high-throughput assays of protein fitness.
`torchdms` provides an interface for fitting regression models to predict the fitness of a protein variant given a one-hot encoded representation of its amino acid sequence.
Specifically `torchdms` uses `PyTorch` [@Paszke2019] for modeling tasks, granting the user a great deal of flexibility when specifying models for their data.
Users can define linear models, arbitrarily deep neural networks, or biophysically motivated models for inferring mutational effects on protein fitness.
`torchdms` will then train these models with specified hyper parameters and produce plots for model evaluation.
This software also allows for users to quickly train and evaluate multiple models with different hyper parameter configurations and is compatible with parallel acceleration via CPU and GPU. Overall, `torchdms` provides researchers with a flexible tool to quickly implement custom models for inferring protein fitness landscapes.



# Deep mutational scanning

Deep mutational scanning (DMS) is an experimental technique that allows researchers to measure the effects of mutations in hundreds of thousands of protein variants in high throughput [@Fowler2014].
Variants assayed in a traditional DMS experiment contain single mutations to a gene of interest, making it relatively straightforward to infer individual mutational effects on a phenotype.
However, recent DMS experiments seek to assay variants containing multiple mutations and infer the nonlinear interactions between mutations made across different genetic background. (Some cites can go here)
These new experiments allow for researchers to study how mutations more generally impact phenotype(s) of a protein and also provide a more vast sampling of the possible sequence space than traditional DMS experiments.
Recently, there have been multiple proposals for inferring mutational effects from DMS data as well as inferring phenotypes for unseen variants.
However, there is still a need for a modeling framework that is able to handle more complicated DMS experiments that measure multiple phenotypes from variants with multiple mutations i.e. [@Starr2020], but also provides the flexibility needed to keep pace with future innovations in DMS.
`torchdms` provides a flexible tool for defining, fitting, and evaluating models for an arbitrary number of phenotypes measured in a DMS experiment.

# Related work

DMS experiments that contain variants with many mutations enable more comprehensive samplings of a fitness landscape but complicate estimation of mutational effects on fitness.
Biologists often desire models with biologically interpretable parameters that explain how certain mutations impact the phenotype being measured in a DMS experiment.
On this front, the proposed model of Global Epistasis (GE) [@Otwinowski2018] infers biologically interpretable mutational effects while also allowing for more complex fitness landscapes to be inferred from DMS data.
However, the original implementation of GE framework presented in [@Otwinowski2018] was provided in the Julia programing language, a powerful but lesser known programming language amongst developers.
MAVE-NN [@Tareen2020] provides a Python implementation of the GE framework with neural networks, but is limited to predicting single phenotypes and by consequence, one set of mutational effects.
In this work, `torchdms` provides a generalized GE framework, that allows for flexible modeling of complex, multi-phenotype DMS experiments, while maintaining the interpretability of inferred mutational effects provided by GE for each phenotype.

# How `torchdms` works
`torchdms` takes the following as input:

-  A pickled object containing a pandas data frame with information about the sequence and fitness of protein variants and a string representing the original protein sequence as input.
- A JSON file containing details for the analysis.
These details include paths to the input file, desired model architecture, as well as specifying any modeling or training hyper parameters such as regularization strength and learning rate.

These inputs are used to create everything needed for the analysis:

- A `SplitDataset` object storing train-validation-test splits alongside their appropriate 1-hot encodings
- A `PyTorch` object with a model architecture, constraints, and hyper parameters as defined in the configuration JSON file
- An `Analysis` object that interfaces contains the data, models and a cost function for model training and evaluation.

After preparing a `SplitDataset` object, users will select an architecture template from `torchdms.models`, details and examples on defining different model templates can be found in the `torchdms` documentation.
`torchdms` then uses the `Analysis` object to train and validate the model, with the resulting model saved as a `torch` object for downstream analysis.
After model training is complete, the model is evaluated on the testing set and plots describing the fit to the testing data are created.
`torchdms` also provides functionality for testing multiple hyper parameter configurations while still defining a single configuration file.
Resulting plots are output to user-specified paths and model predictions on the test set are saved in a CSV file for downstream analysis.

# References
