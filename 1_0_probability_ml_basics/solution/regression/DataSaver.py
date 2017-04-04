import numpy as np
import pickle as pkl
import os


class DataSaver():

  directory = ''
  filename = ''
  path = ''

  def __init__(self, directory, filename):
    self.directory = directory
    self.filename = filename
    self.path = os.path.join(self.directory, self.filename)
    if not os.path.exists(self.directory):
      os.mkdir(self.directory)

  def save_to_file(self, input_data, output_data):
    data = np.hstack([input_data, output_data])
    print('Saving data to relative path {}'.format(self.path))
    pkl.dump(data, open(self.path, 'wb'))

  def restore_from_file(self):
    data = pkl.load(open(self.path, 'rb'))
    input_data = data[:, 0:4]
    output_data = np.reshape(data[:, 4], [data[:, 2].shape[0], 1])
    return input_data, output_data

  def load_submission(self):
    data = pkl.load(open(self.path, 'rb'))
    input_data = data[:, 0:4]
    return input_data
