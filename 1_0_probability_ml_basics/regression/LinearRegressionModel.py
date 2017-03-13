import numpy as np


class ProgrammaticError(Exception):
  """Exception raised when method gets called at a wrong time instance.

  Attributes:
      msg  -- The error message to be displayed.
  """

  def __init__(self, msg):
    self.msg = msg
    print("\033[91mERROR: \x1b[0m {}".format(msg))


class LinearRegressionModel():
  """
  Class for linear regression model.
  """

  # Members
  # Vector in equation: y = F(x,y) * beta, where F(x,y) is the feature vector
  beta = None
  feature_vec = []
  fitting_done = False

  def __init__(self):
    self.feature_vec = []
    self.fitting_done = False

  def error_function(self, predictions, target_values):
    return (predictions - target_values)**2

  def set_feature_vector(self, feature_vec):
    self.feature_vec = feature_vec

  def compute_feature_matrix(self, input_data):
    n_samples = input_data.shape[0]
    n_features = len(self.feature_vec)
    X = np.zeros([n_samples, n_features])
    for i in range(n_samples):
      for j in range(n_features):
        X[i, j] = self.feature_vec[j].evaluate(
            # TODO use all input dimensions
            input_data[i, 0], input_data[i, 1])  # modify this line
    return X

  def fit(self, input_data, output_data):
    n_features = len(self.feature_vec)
    assert n_features > 0, 'Please set the feature vector first.'
    # In general don't use assertions in deployment code, use hard checks, as
    # assertions get compiled away when optimization is used.
    if (n_features < 1):
      raise ProgrammaticError("Please set the feature vector first.")
    n_samples = input_data.shape[0]
    X = self.compute_feature_matrix(input_data)
    # Least squares parameter estimation
    self.beta = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), output_data)
    self.fitting_done = True
    print('Done with fitting of the model.')

  def predict(self, input_data):
    # In general don't use assertions in deployment code, use hard checks, as
    # assertions get compiled away when optimization is used.
    if not self.fitting_done:
      raise ProgrammaticError("Before you use the model for query, you have "
                              "to set the feature vector and fit it.")
    n_features = len(self.feature_vec)
    n_data = input_data.shape[0]
    X = self.compute_feature_matrix(input_data)
    return np.dot(X, self.beta)

  def validate(self, validation_input, validation_output):
    return np.mean(self.error_function(self.predict(validation_input),
                                       validation_output))
