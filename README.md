# Bayesian Optimization of Neural Architectures for Human Activity Recognition
This repository contains code to reproduce experiments used in the HASCA book chapter (to appear soon).

## Description
Design of neural architectures is a critical aspect in deep-learning based methods.
In this chapter, we explore the suitability of different neural architectures for the recognition of mobility-related human activities.
Neural architecture search (NAS) is getting a lot of attention in the machine learning community and improves deep learning models' performances in many tasks like language modeling and image recognition.
Deep learning techniques were successfully applied to human activity recognition (HAR). However, the design of competitive architectures remains cumbersome, time-consuming, and rely strongly on domain expertise.

To address this, we propose a large-scale systematic experimental setup in order to design and evaluate neural architectures for HAR applications.
Specifically, we use a Bayesian optimization (BO) procedure based on a Gaussian process surrogate model in order to tune architectures' hyper-parameters.
We train and evaluate more than 600 different architectures which are then analyzed via the functional ANalysis Of VAriance (fANOVA) framework to assess hyper-parameters relevance.
We experiment our approach on the Sussex-Huawei Locomotion and Transportation (SHL) dataset, a highly versatile, sensor-rich and precisely annotated dataset of human locomotion modes.

## Getting Started

### Prerequisites
* `numpy`
* `TensorFlow`
* `scikit-optimize`
* `fanova` to install, please follow the steps [here](https://automl.github.io/fanova/install.html)

If you are using `pip` package manager, you can simply install all requirements via the following command(s):

    python -m virtualenv .env -p python3 [optional]
    source .env/bin/activate [optional]
    pip3 install -r requirements.txt

### Installing
#### Get the dataset
1. You can get the subset of the SHL dataset (SHL Complete User 1 – Hips phone) from [here](http://www.shl-dataset.org/download/#shldataset-user1-hips). Make sure to put the downloaded files into `./data/` folder.
2. Run `extract_data.sh` script which will extract the dataset into `./generated/tmp/` folder.

## Running
### Bayesian optimization
In order to run Bayesian optimization, you can issue the following command:

    python3 hasca-shl.py --run bayesopt
    
Additionally, you can specify a given type of model or convolutional mode you want to apply Bayesian optimization on as follows:

    python3 hasca-shl.py --run bayesopt --model {cnn|lstm|hybrid}

### Functional analysis of variance
You can find a complete notebook showing the functional analysis of variance inside `notebooks/` folder.

## Authors (Contact)
* Massinissa Hamidi (hamidi@lipn.univ-paris13.fr)
* Aomar Osmani
