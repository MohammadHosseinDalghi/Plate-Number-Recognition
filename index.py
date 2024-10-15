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

# reading images
images_arr = np.empty((0, main_size), dtype=int)
numbers_arr = np.empty((0), dtype=int)

for i in range(1, 10):
    files = os.listdir("dataset/" + str(i))
    for filename in files:
        img = load_img("dataset/" + str(i) + "/" + filename)
        images_arr = np.append(images_arr, [img], axis=0)
        numbers_arr = np.append(numbers_arr, i)

print(numbers_arr)