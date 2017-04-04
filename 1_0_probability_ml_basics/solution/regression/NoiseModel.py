import numpy as np


class NoiseModel():

  mu = 0
  sigma = 0

  def __init__(self, mu=0, sigma=1):
    self.mu = mu
    self.sigma = sigma

  def sample(self, sigma = -1, mu = -1):
    if sigma == -1:
      sigma = self.sigma

    if mu == -1:
      mu = self.mu

    return sigma * np.random.randn() + mu
