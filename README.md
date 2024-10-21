# License Plate Number Recognition using Logistic Regression

This project is designed to detect and recognize numbers from vehicle license plates using Logistic Regression. It leverages OpenCV for image processing, and scikit-learn for machine learning.
  
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
- Python 3.x ( I used 3.8.8)
- OpenCV
- scikit-learn
- Numpy
- Matplotlib

You can install the required libraries using:
```bash
pip install opencv-python
pip install scikit-learn
pip install numpy
pip install matplotlib
```

## Data Structure
The project assumes that the dataset is structured as follows:
```bash
dataset/
│
├── 1/     # Folder containing images of the digit 1
├── 2/     # Folder containing images of the digit 2
├── ...
├── 9/     # Folder containing images of the digit 9
```
Each folder contains multiple images of the respective digit for training the model.

## How It Works:

1. Image Preprocessing:
    - License plate images are resized to a standard size of 8x32 pixels and flattened.
    - The plate image is converted to grayscale and blurred to reduce noise.
    - A binary threshold is applied to highlight the digits on the plate.
2. Model Training:
    - The digit dataset is loaded and used to train a logistic regression model. Each digit is flattened into a feature vector.
    - The model is trained using the `train_test_split` function from scikit-learn, with an 80/20 split between training and test sets.
3. Number Detection:
    - The code analyzes pixel intensity across columns to locate the boundaries of each digit on the plate.
    - If a significant change in intensity is detected (based on a threshold), the region is isolated as a potential digit.
4. Prediction:
    - The extracted digit region is resized to the standard shape and flattened.
    - The trained logistic regression model predicts the digit.
    - The final number is displayed on the image.
  
## Usage
1. Clone the repository and navigate to the project directory:
```bash
git clone https://github.com/MohammadHosseinDalghi/Plate-Number-Recognition
cd Plate-Number-Recognition
```
2. Ensure you have the correct dataset structure as described above.
3. Run the project:
```bash
python index.py
```
4. Provide the license plate image in the script, and the model will predict the numbers.

## Results
The predicted license plate number is printed on the console and displayed on the image. Each detected digit is visualized with a bounding box on the plate.
