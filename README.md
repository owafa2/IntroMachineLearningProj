# IntroMachineLearningProj

## Team: The Debuggers

## Overview
This project compares two deep learning models on the CIFAR-10 image classification task:

- SmallCNN trained from scratch  
- Fine-tuned ResNet-18  

The goal is to classify 32x32 color images into one of 10 object categories:
airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.

---

## Abstract
This project explores image classification using the CIFAR-10 dataset. Two models were implemented and compared: a custom SmallCNN and a fine-tuned ResNet-18. Both models were trained and evaluated on the same dataset with standard preprocessing techniques. The results show that ResNet-18 significantly outperforms SmallCNN, achieving an accuracy of 96.22% compared to 91.02%. Confidence intervals and normalized confusion matrices were used to evaluate model performance. Based on these results, ResNet-18 is selected as the best model due to its higher accuracy and overall performance.

---

## Repository Structure
- `cifar10.py` – main training and evaluation script  
- `data/cifar-10-batches-py/` – CIFAR-10 dataset files  
- `outputs/metrics.json` – final evaluation metrics for both models  
- `outputs/confmat_smallcnn.png` – normalized confusion matrix for SmallCNN  
- `outputs/confmat_resnet18.png` – normalized confusion matrix for ResNet-18  
- `outputs/curves_smallcnn.png` – training and validation curves for SmallCNN  
- `outputs/curves_resnet18.png` – training and validation curves for ResNet-18  

---

## Problem Description
This project addresses multi-class image classification on the CIFAR-10 dataset.

### Input
A 32x32 RGB image.

### Output
One of 10 class labels:
- airplane  
- automobile  
- bird  
- cat  
- deer  
- dog  
- frog  
- horse  
- ship  
- truck  

---

## Data Preprocessing
Two preprocessing pipelines were used:

### SmallCNN
- random horizontal flip  
- random crop with padding  
- tensor conversion  
- CIFAR-10 normalization  

### ResNet-18
- resize to 224x224  
- random horizontal flip  
- random crop  
- tensor conversion  
- ImageNet normalization  

The training set was split into:
- 45,000 training images  
- 5,000 validation images  

The official CIFAR-10 test set was used for final evaluation.

---

## Models

### 1. SmallCNN
A custom convolutional neural network with three convolutional blocks followed by a classifier layer. This model is trained from scratch on the CIFAR-10 dataset.

### 2. ResNet-18
A pretrained ResNet-18 model that was fine-tuned on CIFAR-10 by replacing the final fully connected layer with a 10-class output layer. Transfer learning allows this model to achieve better performance.

---

## Performance Results
Final test results from `outputs/metrics.json`:

### SmallCNN
- Accuracy: **0.9102**  
- 95% CI: **[0.9044, 0.9153]**  
- Macro Precision: **0.9099**  
- Macro Recall: **0.9102**  
- Macro F1: **0.9100**  

### ResNet-18
- Accuracy: **0.9622**  
- 95% CI: **[0.9585, 0.9660]**  
- Macro Precision: **0.9623**  
- Macro Recall: **0.9622**  
- Macro F1: **0.9622**  

---

## Model Comparison
ResNet-18 outperforms SmallCNN across all evaluation metrics. It achieves higher accuracy, precision, recall, and F1-score. This improvement is mainly due to transfer learning, which allows ResNet-18 to leverage features learned from a large dataset.

SmallCNN, while simpler and faster to train, is less effective at capturing complex patterns in image data. Therefore, ResNet-18 is the better model for this task.

---

## Best Model
ResNet-18 performed better than SmallCNN on all reported metrics. It achieved the highest test accuracy and macro F1 score, making it the best overall model in this project.

---

## Confidence Intervals
This project includes 95% bootstrap confidence intervals for test accuracy for both models, providing a reliable measure of performance variability.

---

## Confusion Matrices
Normalized confusion matrices for both models are included in the `outputs/` folder to visualize classification performance across all classes.

---

## Pros and Cons

### Pros
- ResNet-18 achieves high accuracy and strong generalization  
- Transfer learning significantly improves performance  
- Confidence intervals provide robust evaluation  
- Normalized confusion matrices offer detailed insights  

### Cons
- ResNet-18 requires more computational resources  
- SmallCNN underperforms compared to deeper models  
- CIFAR-10 images are low resolution  
- Deep learning models are less interpretable  

---

## How to Run

```bash
python cifar10.py --epochs_cnn 30 --epochs_resnet 15
