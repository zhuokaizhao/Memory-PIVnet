#!/bin/bash
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -y
conda install h5py -y
conda install -c conda-forge tqdm -y
conda install -c conda-forge matplotlib -y
conda install -c anaconda pillow -y
conda install -c anaconda seaborn -y
pip install flowiz -U
