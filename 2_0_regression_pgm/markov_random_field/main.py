#!/usr/bin/env python

import pickle as pkl
import numpy as np
import MarkovRandomField
import cv2

image1 = pkl.load(open("data_1.pkl", 'rb'))
image2 = pkl.load(open("data_2.pkl", 'rb'))
features1 = cv2.cvtColor(image1, cv2.COLOR_BGR2LAB)
features2 = cv2.cvtColor(image2, cv2.COLOR_BGR2LAB)

# TODO: Tune the parameters
beta = 1  # strength of the second order clique criteria
threshold = 0  # convergence threshold
max_iterations = 10  # maximum number of iterations
temperature_init = 5  # initial temperature
discount = 1  # temperature scheduler's factor

number_of_labels = 6  # number of labels in the image
size_ratio = 4  # increases the image by size_ratio, just for displaying

image_segmentation1 = MarkovRandomField.ImageSegmenter(
    beta, threshold, max_iterations, temperature_init, discount,
    number_of_labels, size_ratio, features1, image1)
result1 = image_segmentation1.segmentImage()

# TODO: Tune the parameters
beta = 1  # strength of the second order clique criteria
threshold = 0  # convergence threshold
max_iterations = 10  # maximum number of iterations
temperature_init = 5  # initial temperature
discount = 1  # temperature scheduler's factor

number_of_labels = 3  # number of labels in the image
size_ratio = 4  # increases the image by size_ratio, just for displaying

image_segmentation2 = MarkovRandomField.ImageSegmenter(
    beta, threshold, max_iterations, temperature_init, discount,
    number_of_labels, size_ratio, features2, image2)
result2 = image_segmentation2.segmentImage()
