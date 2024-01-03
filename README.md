# MNIST Neural Network Classification

This project demonstrates the training and evaluation of neural network models for the classification of handwritten digits using the MNIST dataset. It includes both a fully connected neural network and a convolutional neural network (CNN) model for the task.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Getting Started](#getting-started)
- [Training and Evaluation](#training-and-evaluation)
- [Results](#results)

## Overview

The MNIST dataset consists of 28x28 pixel grayscale images of handwritten digits (0-9) and is a popular benchmark dataset in the field of computer vision and deep learning. This project explores the training and evaluation of neural network models to classify these digits.

## Prerequisites

Before running the code in this project, ensure you have the following dependencies installed:

- Python (>=3.6)
- PyTorch
- torchvision
- numpy
- matplotlib
- scikit-learn
- seaborn (for visualization)


## Getting Started

1. Clone this GitHub repository to your local machine:

```bash
git clone https://github.com/yourusername/mnist-neural-network-classification.git
```

2. Navigate to the project directory:

```bash
cd mnist-neural-network-classification
```

3. Run the provided Python script to train and evaluate the models:

```bash
python mnist_classification.py
```

4. You can customize the hyperparameters, model architecture, and other settings within the `mnist_classification.py` script to experiment with different configurations.

## Training and Evaluation

The project includes two types of models:

- Fully Connected Model: This model consists of multiple fully connected (dense) layers with dropout and uses the Negative Log Likelihood (NLL) loss function with Stochastic Gradient Descent (SGD) as the optimizer.

- Convolutional Neural Network (CNN) Model: This model utilizes convolutional layers followed by a linear layer and is trained with the Cross-Entropy loss function and the Adam optimizer.

Both models are trained on the MNIST training dataset and evaluated on the test dataset.

## Results

After running the training and evaluation script, you will obtain results such as:

- Loss during training
- Test loss
- Accuracy

The results provide insights into the performance of the models in classifying the handwritten digits.
