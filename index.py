# Importing libraries
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split

# set the main size
standard_shape = (8, 32)
main_size = standard_shape[0] * standard_shape[1]
# print(main_size)
box = np.empty((0, 4), dtype=int)
plate_name = "plate.jpg"

# Define load_img function
def load_img(image):
    img = cv2.resize(cv2.imread(image, 0), (standard_shape[0], standard_shape[1])).flatten()
    return img
