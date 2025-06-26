# seismic-autoencoder
Reading SEGY seismic data, patch extraction, and ML preparation
# Seismic Autoencoder: SEGY Data Reading and Patch Extraction

This repository contains Python code for reading seismic data in SEGY format, extracting 64x64 patches, and preparing the data for machine learning tasks such as anomaly detection with autoencoders.

## Project Overview

-  Read SEGY seismic data using the `segyio` library  
-  Extract patches of size 64x64 from seismic data for ML input  
-  Visualize seismic data and extracted patches using `matplotlib`  
-  Prepare groundwork for building convolutional autoencoder models  

## Dataset

The SEAM Phase I dataset is used, which is a publicly available synthetic seismic volume developed for geophysical research.

## Installation

To install the required Python packages, run:

```bash
pip install numpy matplotlib segyio tensorflow scikit-learn
