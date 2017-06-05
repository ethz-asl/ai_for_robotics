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
    return 1/(1+np.exp(-np.dot(w.transpose(), x.transpose())))


# To make it easier the 24x24 pixels have been reshaped to a vector of 576 pixels. the value corrsponds to the greyscale intensity of the pixel
input_data = np.genfromtxt(open("XtrainIMG.txt"))  # This is an array that has the features (all 576 pixel intensities) in the columns and all the available pictures in the rows
output_data = np.genfromtxt(open("Ytrain.txt"))  # This is a vector that has the classification (1 for open eye 0 for closed eye) in the rows

n_samples = input_data.shape[0]
n_features = input_data.shape[1]

# split data into training and validation data
ratio_train_validate = 0.8
idx_switch = int(n_samples * ratio_train_validate)
training_input = input_data[:idx_switch, :]
training_output = output_data[:idx_switch][:,None]
validation_input = input_data[idx_switch:, :]
validation_output = output_data[idx_switch:][:,None]

#TODO initialise w
w = 0.01*np.random.normal(size = [n_features, 1])

#TODO implement the iterative calculation of w
#TODO2: modify the algorithm to account for regularization as well to improve the classifier
lambda_v = 0.1
X= training_input
y = training_output
q=np.zeros([X.shape[0], 1])
mu = np.zeros([X.shape[0], 1])


for j in range(0,9):
 for i in range(0,X.shape[0]):
   q[i]=logistic_function(w,X[i,:])*(1-logistic_function(w,X[i,:]))
   mu[i]=logistic_function(w,X[i,:])
 helper1 = np.linalg.inv(np.dot(np.dot(X.transpose(),np.diag(q[:,0])),X)+lambda_v*np.eye(X.shape[1]))
 #helper2 = np.dot(np.dot(np.diag(q[:,0]),X),w) + y - mu + lambda_v*np.linalg.norm(w)
 helper2 = np.dot(X.transpose(), (mu - y)) + lambda_v*w
 w_new = w - np.dot(helper1,helper2)
 print np.linalg.norm(y-mu)
 w = w_new

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
