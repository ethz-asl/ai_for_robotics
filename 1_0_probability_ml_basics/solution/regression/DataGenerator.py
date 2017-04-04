import NoiseModel as noise
import Features as features
import DataSaver as saver

import numpy as np
import pickle as pkl
import os


class DataGenerator():
  """
  Data generation for linear regression example.
  """

  feature_vec = []
  feature_weights = []
  noise_model = noise.NoiseModel(0, 8)
  input_data = np.zeros([0, 5])
  output_data = np.zeros([0, 1])
  max_x1 = 0
  max_x2 = 0
  max_x3 = 0
  max_x4 = 0

  saver = None

  def __init__(self):
    self.feature_vec = [features.CrossTermX1X3(), features.SinX2(),
                        features.SquareX4(), features.Identity()]
    self.feature_weights = [0.1, -2, -0.3, 3]
    self.noise_model = noise.NoiseModel()
    self.max_x1 = 10
    self.max_x2 = 10
    self.max_x3 = 10
    self.max_x4 = 10
    self.saver = saver.DataSaver('data', 'submission_data.pkl')

  def __generate_samples__(self, n_samples):
    input_x1 = np.random.rand(n_samples) * self.max_x1
    input_x2 = np.random.rand(n_samples) * self.max_x2
    input_x3 = np.random.rand(n_samples) * self.max_x3
    input_x4 = np.random.rand(n_samples) * self.max_x4
    self.input_data = np.array([input_x1, input_x2, input_x3, input_x4]).T
    self.output_data = np.zeros([n_samples, 1])

    for i in range(n_samples):
     out_value = 0
      for f, w in zip(self.feature_vec, self.feature_weights):
       out_value += w * \
           f.evaluate(self.input_data[i, 0], self.input_data[i, 1], self.input_data[i, 2], self.input_data[i, 3])

     out_value += self.noise_model.sample()
      self.output_data[i] = out_value

    return self.input_data, self.output_data

  def sample(self, n_samples):
    if self.output_data.shape[0] == n_samples:
      print('Returning pre-generated samples.')
     return self.input_data, self.output_data
    else:
      print('Generating new samples.')
      return self.__generate_samples__(n_samples)

  def save(self):
    self.saver.save_to_file(self.input_data), self.output_data)
