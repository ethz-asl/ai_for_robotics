####################################
# Author: Mark Pfeiffer            #
# Date created: 30.05.2017         #
#                                  #
# Date last changed: 11.10.2017    #
# Changed by: Mark Pfeiffer        #
####################################


import Support as sup
import numpy as np

class GradientDescentOptimizer():
  """
  Gradient descent optimization for neural network parameters.
  """
  learning_rate = 0

  def __init__(self, learning_rate=0.01):
    self.learning_rate = learning_rate

  def getUpdatedParameters(self, nn, gradients):
    """
    Update parameters of the network and return them.
    """
    #TODO: update network parameters 
    
  def computeBatchGradient(self, gradient_list):
    """
    Compute the gradient for a whole data batch from a provided gradient list. 
    Input: 
      Gradient list contains the gradient for each sample in the data batch. The structure is a list of variables (provided data structure support.Variable()). 
      The weights and biases members both contain the gradients of all the layers for one data sample.
    Return value: 
      One fused gradient including all data sample gradients.
    """
    batch_gradient = gradient_list[0]
    for g in gradient_list[1:]:
      batch_gradient = batch_gradient + g
    return batch_gradient

  def updateStep(self, nn, loss_function, x_batch, y_target_batch):
    """
    Update the NN model parameters given the loss function and a data batch.
    """
    gradients = []
    avg_batch_loss = 0
    batch_size = x_batch.shape[0]
    
    for i in range(x_batch.shape[0]):
      x = np.array([x_batch[i,:]])
      y_target = np.array([y_target_batch[i,:]])
      y = nn.output(x)
      avg_batch_loss += loss_function.evaluate(y, y_target)
      nn_gradient = nn.gradients(x, loss_function, y_target)
      gradients.append(nn_gradient)
    
    batch_gradient = self.computeBatchGradient(gradients)
    new_p = self.getUpdatedParameters(nn, batch_gradient)
    nn.setParameters(new_p)
    return avg_batch_loss/batch_size

