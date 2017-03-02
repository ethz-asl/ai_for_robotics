import numpy as np


class NoiseModel():

  mu = 0
  sigma = 0

  def __init__(self, mu=0, sigma=1):
    self.mu = mu
    self.sigma = sigma

  def sample(self):
    #TODO: return Gaussian noise sample
