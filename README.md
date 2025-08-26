# USTDFuse in PyTorch
Implementation of "USTDFuse:Structure-Texture Decomposition based deep unrolling networks for Infrared-Visible Image Fusion" in PyTorch.


# Requirements
# create & activate
conda create -n tsdfuse python=3.8 -y
conda activate tsdfuse
# install deps
pip install -r requirements.txt 

# DATA

# Test
python test.py

# Train
python decom-train.py
python decom-train-jixu.py
python train.py
