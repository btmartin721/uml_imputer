#!/bin/bash

conda create -n uml_imputer_dev python=3.8
conda activate uml_imputer_dev

conda install matplotlib seaborn jupyterlab scikit-learn=1.0 tqdm pandas=1.2.5 numpy=1.20.2 scipy=1.6.2

# Only works if using Intel CPUs; speeds up processing
conda install scikit-learn-intelex

conda install -c conda-forge toytree kaleido

# For PCA plots.
conda install -c plotly plotly

# For genetic algorithm plotting functions
pip install sklearn-genetic-opt[all]

pip install scikeras

pip install tensorflow-cpu==2.7

pip install black
