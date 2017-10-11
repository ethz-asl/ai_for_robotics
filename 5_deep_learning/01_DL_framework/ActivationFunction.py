import numpy as np
from abc import ABCMeta, abstractmethod

class ActivationFunction():
  """
  Activation function base class.
  """
  __metaClass__ = ABCMeta

  @abstractmethod
  def evaluate(self, input):
    pass

  @abstractmethod
  def derivative(self, input):
    pass


class SigmoidActivation(ActivationFunction):
  """
  Sigmoid activation function. Sigmoid works elementwise on an array.
  """
  def evaluate(self, input):
    return 1 / (1 + np.exp(-input))

  def derivative(self, input):
    return np.multiply(self.evaluate(input), (1 - self.evaluate(input)))  # dotwise multiplication


class UnitActivation(ActivationFunction):
  """
  A unit activation function.
  This evaluates to the input and its derivative is one.
  """
  def evaluate(self, input):
    return input

  def derivative(self, input):
    return np.ones(input.shape)
