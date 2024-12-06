# Simple Quantization Example with PyTorch

This project demonstrates how to implement quantization using PyTorch. Quantization is a technique used to reduce the precision of the weights and activations in a model, which can lead to smaller model sizes, faster inference, and lower memory usage.

## Overview

This project consists of a simple feedforward neural network trained on the MNIST dataset. After training, the model is quantized using PyTorch's quantization API. The project demonstrates the following steps:

1. **Model Definition**: A simple neural network is defined.
2. **Data Preparation**: The MNIST dataset is loaded and transformed.
3. **Training**: The model is trained for a few epochs.
4. **Evaluation**: The model is evaluated on the test set before and after quantization.
5. **Quantization**: Post-training quantization is applied to the model.
6. **Model Size Comparison**: The sizes of the original and quantized models are compared.

## Requirements

- Python 3.6+
- PyTorch
- torchvision

You can install the required packages using pip:

```bash
pip install torch torchvision
