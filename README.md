## FEIIDS: Feature Engineering-Based Intrusion Detection System for IoT Security

This repository contains Machine Learning (ML) and Deep Learning (DL) implementations for intrusion detection using the UNSW-NB15 dataset. It includes complete workflows for data preprocessing, feature subsetting, regression-based feature engineering, model training, evaluation, and result comparison.

# Dataset

Download the UNSW-NB15 dataset from the link below:
 https://www.kaggle.com/datasets/sadhwanisapna/unsw-nb15-feiids-test-and-train

Contains the original dataset, processed feature-engineered dataset, and train/test splits used in the study.

# Implementations Included
Machine Learning Models

Decision Tree

Logistic Regression

Gaussian Naive Bayes

Random Forest

Deep Learning Model

BiLSTM with early stopping

# Workflow

Load and preprocess dataset (cleaning, encoding, normalization)

Extract feature subsets using Association Rule Mining

Apply binary label encoding per attack class

Generate regression-based probability scores (feature engineering)

Train ML/DL models and evaluate with:

Accuracy

Precision

Recall

F1 Score

Confusion Matrix

# Key Results

Binary Accuracy: up to 98.80% (RF)

Multiclass Accuracy: up to 97.03% (RF)

Training Time: ~280s for 2M records

Deep models require ~1900s for just 14K records

FEIIDS delivers lightweight, high-performance intrusion detection suitable for real-time IoT environments.
