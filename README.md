# Telecommunications Lab – Automatic Modulation Classification

## Project Overview

This project implements an **Automatic Modulation Classification (AMC)** system using Machine Learning techniques. 
The goal is to identify the modulation type of a received wireless signal under noisy and multipath channel conditions.

The system classifies the following digital modulation schemes:

- QPSK
- 8PSK
- 16QAM
- 64QAM

---

## System Pipeline

The implemented pipeline consists of the following steps:

1. **Signal Generation**
   - Synthetic generation of digital modulated signals

2. **Channel Modeling**
   - Multipath fading channel simulation
   - AWGN (Additive White Gaussian Noise)

3. **Feature Extraction**
   Extracted statistical features such as:
   - Mean, Variance
   - Skewness, Kurtosis
   - Amplitude & Phase statistics
   - Power-related metrics
   - Higher-order moments

4. **Machine Learning Models**
   - K-Nearest Neighbors (KNN)
   - Support Vector Machine (SVM)
   - Random Forest (RF)
   - Ensemble Voting Classifier

5. **Evaluation Metrics**
   - Accuracy
   - F1-score (macro)
   - Cohen’s Kappa
   - Matthews Correlation Coefficient (MCC)
   - Confusion Matrix

---

## Machine Learning Approach

The classification is performed using supervised learning. 
Feature vectors extracted from the received signal are used to train classifiers to distinguish between modulation types.

An ensemble method (Voting Classifier) is also used to improve robustness and performance.

---

## Performance Evaluation

The models are evaluated under different **SNR (Signal-to-Noise Ratio)** levels:

- 0 dB
- 5 dB
- 10 dB
- 15 dB
- 20 dB

This allows analysis of model robustness under noisy conditions.

---



