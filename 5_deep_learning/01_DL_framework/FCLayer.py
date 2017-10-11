####################################
# Author: Mark Pfeiffer            #
# Date created: 30.05.2017         #
#                                  #
# Date last changed: 11.10.2017    #
# Changed by: Mark Pfeiffer        #
####################################

import numpy as np


class FCLayer():
  """
  A fully connected layer.
  """

  activation_function = None
  W = None
  b = None
  layer_number = None

  def __init__(self, layer_number, activation_function, input_size, output_size):
    self.layer_number = layer_number
    self.activation_function = activation_function
    self.W = np.random.randn(input_size, output_size)
    self.b = np.random.randn(1, output_size)

  def affinePart(self, x):
    '''
    Get the affine part of the layer, excluding the activation function.
    '''
    return #TODO (output dimension should be a row vector)

  def output(self, x):
    '''
    Layer output. Activation function applied to affine part.
    '''
    return #TODO

  def derivativeActivation(self, x):
    '''
    Derivative of the activation function of this layer. 
    Will be required by the chain rule.
    '''
    return self.activation_function.derivative(self.affinePart(x))

  def inputDimension(self):
    return self.W.shape[0]

  def outputDimension(self):
    return self.W.shape[1]

  # Getters and setters
  def setWeights(self, W):
    assert self.W.shape == W.shape, 'Weight shapes are inconsistent.'
    self.W = W

  def getWeights(self):
    return self.W

  def setBiases(self, b):
    assert self.b.shape == b.shape, 'Bias shapes are inconsistent.'
    self.b = b

  def getBiases(self):
    return self.b

  def __len__(self):
    return self.outputDimension()
