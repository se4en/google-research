#!/usr/bin/env bash
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b -p
rm ~/miniconda.sh
source $HOME/miniconda3/bin/activate
conda update -n base -c defaults conda
conda install -c conda-forge lightfm
conda install -c anaconda pandas
conda install -c intel scikit-learn
conda install -c conda-forge hydra-core