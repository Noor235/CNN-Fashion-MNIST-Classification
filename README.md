# Deep Learning-Based Image Classification Using CNN

## Overview
This project implements and compares three CNN architectures 
for classifying fashion items from the Fashion-MNIST dataset 
using PyTorch.

## Dataset
- **Name:** Fashion-MNIST
- **Source:** https://www.kaggle.com/datasets/zalando-research/fashionmnist
- **Total Images:** 70,000 grayscale images
- **Classes:** 10 fashion categories
- **Image Size:** 28×28 pixels

## Models
| Model | Architecture | Accuracy |
|-------|-------------|----------|
| Model 1 | Basic CNN 2 Conv Layers | 89.41% |
| Model 2 | CNN 4 Conv Layers | 90.37% |
| Model 3 | Deep CNN 5+ Conv Layers | 86.05% |

## Results
- Best Model: **Model 2** with 90.37% accuracy
- Macro Average AUC: **0.9647**
- Overall F1-Score: **0.87**

## Technologies
- Python 3.13
- PyTorch
- Scikit-learn
- Matplotlib
- Seaborn
- Pandas
- NumPy
