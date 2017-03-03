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
  noise_model = None
  input_data = np.zeros([0, 2])
  output_data = np.zeros([0, 1])
  max_x1 = 0
  max_x2 = 0
  saver = None

  def __init__(self):
    self.feature_vec = [features.LinearX1(), features.LinearX2(),
                        features.SquareX1(), features.ExpX2(), 
                        features.LogX1(), features.Identity()]
    self.feature_weights = [1, 2, 1, 0.1, 10, 40]
    self.noise_model = noise.NoiseModel()
    self.max_x1 = 10
    self.max_x2 = 10
    self.saver = saver.DataSaver('data', 'data_samples.pkl')

  def __generate_samples__(self, n_samples):
    input_x1 = np.random.rand(n_samples) * self.max_x1
    input_x2 = np.random.rand(n_samples) * self.max_x2
    self.input_data = np.array([input_x1, input_x2]).T
    self.output_data = np.zeros([n_samples, 1])

    for i in range(n_samples):
      out_value = 0
      for f, w in zip(self.feature_vec, self.feature_weights):
        out_value += w * \
            f.evaluate(self.input_data[i, 0], self.input_data[i, 1])
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
    self.saver.save_to_file(self.input_data, self.output_data)
