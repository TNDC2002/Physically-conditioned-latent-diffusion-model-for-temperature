#!/bin/bash

# download data
wget "https://zenodo.org/records/12941117/files/pretrained_models.zip?download=1" -O pretrained_models.zip

# unzip data
unzip pretrained_models -d ./
rm pretrained_models.zip

# PDE Loss Model 
wget "https://zenodo.org/records/15460090/files/pde_loss_model_checkpoint.ckpt?download=1" -O pde_loss_model_checkpoint.zip
unzip pde_loss_model_checkpoint.zip
rm pde_loss_model_checkpoint.zip

