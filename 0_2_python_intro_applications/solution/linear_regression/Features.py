from abc import ABCMeta, abstractmethod
import math

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
    return x1


class LinearX2(Feature):

  def evaluate(self, x1, x2):
    return x2


class SquareX1(Feature):

  def evaluate(self, x1, x2):
    return x1**2


class SquareX2(Feature):

  def evaluate(self, x1, x2):
    return x2**2


class CrossTermX1X2(Feature):

  def evaluate(self, x1, x2):
    return x1 * x2
  
class ExpX1(Feature):
  
  def evaluate(self, x1, x2):
    return math.exp(x1)

class ExpX2(Feature):
  
  def evaluate(self, x1, x2):
    return math.exp(x2)
  
class LogX1(Feature):
  
  def evaluate(self, x1, x2):
    return math.log(x1)
  
class LogX2(Feature):
  
  def evaluate(self, x1, x2):
    return math.log(x2)

class Identity(Feature):

  def evaluate(self, x1, x2):
    return 1
