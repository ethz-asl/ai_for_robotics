#!/usr/bin/env python

import Features as features
import LinearRegressionModel as model
import DataSaver as saver
import matplotlib.pyplot as plt
import pickle as pkl

# Remove lapack warning on OSX (https://github.com/scipy/scipy/issues/5998).
import warnings
warnings.filterwarnings(action="ignore", module="scipy",
                        message="^internal gelsd")

plt.close('all')

# TODO decide if you want to show the plots to compare input and output data
show_plots = False

# data_generator = data.DataGenerator()
data_saver = saver.DataSaver('data', 'data_samples.pkl')
input_data, output_data = data_saver.restore_from_file()
n_samples = input_data.shape[0]
if(show_plots):
  plt.figure(0)
  plt.scatter(input_data[:, 0], output_data[:, 0])
  plt.xlabel("x1")
  plt.ylabel("y")
  plt.figure(1)
  plt.scatter(input_data[:, 1], output_data[:, 0])
  plt.xlabel("x2")
  plt.ylabel("y")
  if (input_data.shape[1] > 2):
    plt.figure(2)
    plt.scatter(input_data[:, 2], output_data[:, 0])
    plt.xlabel("x3")
    plt.ylabel("y")
    plt.figure(3)
    plt.scatter(input_data[:, 3], output_data[:, 0])
    plt.xlabel("x4")
    plt.ylabel("y")


# Split data into training and validation
# TODO Overcome the problem of differently biased data
ratio_train_validate = 0.8
idx_switch = int(n_samples * ratio_train_validate)
training_input = input_data[:idx_switch, :]
training_output = output_data[:idx_switch, :]
validation_input = input_data[idx_switch:, :]
validation_output = output_data[idx_switch:, :]

# Fit model
lm = model.LinearRegressionModel()
# TODO use and select the new features
lm.set_feature_vector([features.LinearX1(), features.LinearX2(),
                       features.SquareX1(), features.ExpX2(),
                       features.LogX1(), features.Identity()])
lm.fit(training_input, training_output)


# Validation
mse = lm.validate(validation_input, validation_output)
print('MSE: {}'.format(mse))
print(' ')
print('feature weights \n{}'.format(lm.beta))

# load submission data
submission_loader = saver.DataSaver('data', 'submission_data.pkl')
submission_input = submission_loader.load_submission()

# predict output
submission_output = lm.predict(submission_input)

#save output
pkl.dump(submission_output, open("results.pkl", 'wb'))

plt.show()
