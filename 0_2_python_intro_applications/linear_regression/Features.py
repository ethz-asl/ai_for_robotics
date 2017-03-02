import numpy as np
from abc import ABCMeta, abstractmethod


class Feature():
  """
  Feature base class.
  """
  __metaClass__ = ABCMeta

  @abstractmethod
  def evaluate(self, x1, x2):
    pass


# Feature classes
class LinearX1(Feature):

  def evaluate(self, x1, x2):
    #TODO
    
  
class Identity(Feature):

  def evaluate(self, x1, x2):
    return 1
