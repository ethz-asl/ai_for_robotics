#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 12:13:01 2017

@author: sebastian
"""
import numpy as np
import matplotlib.pyplot as plt


def logistic_function(w, x):
    # TODO implement the logistic function
    return np.dot(w.transpose(), x.transpose()) # change this line


# To make it easier the 24x24 pixels have been reshaped to a vector of 576 pixels. the value corrsponds to the greyscale intensity of the pixel
input_data = np.genfromtxt(open("XtrainIMG.txt"))  # This is an array that has the features (all 576 pixel intensities) in the columns and all the available pictures in the rows
output_data = np.genfromtxt(open("Ytrain.txt"))  # This is a vector that has the classification (1 for open eye 0 for closed eye) in the rows

n_samples = input_data.shape[0]
n_features = input_data.shape[1]


ratio_train_validate = 0.8
idx_switch = int(n_samples * ratio_train_validate)
training_input = input_data[:idx_switch, :]
training_output = output_data[:idx_switch][:,None]
validation_input = input_data[idx_switch:, :]
validation_output = output_data[idx_switch:][:,None]

#TODO initialise w
w = 0 # change this line

#TODO implement the iterative calculation of w
#TODO2: modify the algorithm to account for regularization as well to improve the classifier

#validation
h = logistic_function(w,validation_input)
output = np.round(h).transpose()

error = np.abs(output-validation_output).sum()

print 'wrong classification of ',(error/output.shape[0]*100),'% of the cases in the validation set'


# classify test data for evaluation
test_input = np.genfromtxt(open("XtestIMG.txt"))
h = logistic_function(w,test_input)
test_output = np.round(h).transpose()
np.savetxt('results.txt', test_output)
