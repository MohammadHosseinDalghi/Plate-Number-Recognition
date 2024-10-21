# License Plate Number Recognition using Logistic Regression

This project is designed to detect and recognize numbers from vehicle license plates using Logistic Regression. It leverages OpenCV for image processing, and scikit-learn for machine learning.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Dataset Structure](#dataset-structure)
- [How it Works](#how-it-works)
- [Usage](#usage)
- [Results](#results)
- [Future Improvements](#future-improvements)
- [Contributors](#contributors)
  
## Overview
This project uses a machine learning model to recognize digits from images of license plates. The goal is to extract individual numbers from the plate image and predict the digits using logistic regression.

The workflow involves:
1. Preprocessing plate images.
2. Training a logistic regression model on the digit dataset.
3. Detecting digit locations on the plate image.
4. Predicting digits and displaying the final plate number.

## Features
- Reads and preprocesses license plate images.
- Trains a logistic regression model to recognize digits from images.
- Detects number regions on the plate using pixel intensity.
- Visualizes the detected numbers on the plate image.
- Predicts and prints the final plate number.

## Requirements
- Python 3.x
- OpenCV
- scikit-learn
- Numpy
- Matplotlib

You can install the required libraries using:
```bash
pip install -r requirements.txt
