#!/usr/bin/env bash
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b -p
rm ~/miniconda.sh
source $HOME/miniconda3/bin/activate
conda update -n base -c defaults conda
conda install -c -y conda-forge lightfm
conda install -c -y anaconda pandas
conda install -c -y intel scikit-learn
conda install -c -y conda-forge hydra-core