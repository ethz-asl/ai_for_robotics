#!/usr/bin/env python

import numpy as np
import NoiseModel as noise
import Features as features
import LinearRegressionModel as model
import DataSaver as saver
import pylab as pl
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression

# Remove lapack warning on OSX (https://github.com/scipy/scipy/issues/5998).
import warnings
warnings.filterwarnings(action="ignore", module="scipy",
                        message="^internal gelsd")

pl.close('all')

# data_generator = data.DataGenerator()
data_saver = saver.DataSaver('data', 'data_samples.pkl')
input_data, output_data = data_saver.restore_from_file()
n_samples = input_data.shape[0]

# Split data into training and validation
ratio_train_validate = 0.8
idx_switch = int(n_samples * ratio_train_validate)
training_input = input_data[:idx_switch, :]
training_output = output_data[:idx_switch, :]
validation_input = input_data[idx_switch:, :]
validation_output = output_data[idx_switch:, :]

# Fit model
lm = model.LinearRegressionModel()
lm.set_feature_vector([features.LinearX1(), #TODO ])
lm.fit(training_input, training_output)

# sklearn
#TODO

# Validation
mse = lm.validate(validation_input, validation_output)
print('MSE: {}'.format(mse))
print(' ')
print('feature weights \n{}'.format(lm.beta))
validation_predictions = lm.predict(validation_input)
feature_matrix_validation = lm.compute_feature_matrix(validation_input)
validation_predictions_sk = linear_regression_sk.predict(
    feature_matrix_validation)

# Visualization
fig = pl.figure()
ax = fig.add_subplot(121, projection='3d')
ax.scatter(training_input[:, 0], training_input[:, 1], training_output,
           alpha=1)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('y')
ax.set_title('Training data')

ax = fig.add_subplot(122, projection='3d')
ax.scatter(validation_input[:, 0], validation_input[:, 1],
           validation_output, alpha=0.5, c='b')
ax.scatter(validation_input[:, 0], validation_input[:, 1],
           validation_predictions, alpha=0.5, c='r')
# ax.scatter(validation_input[:, 0], validation_input[:, 1],
#            validation_predictions_sk, alpha=0.5, c='g')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('y')
ax.set_title('Validation results')

pl.show()
