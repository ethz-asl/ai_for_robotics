#!/usr/bin/env python

import sys
sys.path = ['/usr/local/lib/python2.7/dist-packages'] + sys.path

import unittest
import numpy as np

import ActivationFunction as activation
import GradientDescentOptimizer as gdo
import FCLayer as layer
import FCNetwork as network
import LossFunction as loss
import Support as sup

# TODO: Update and extend tests

# Global variables for test network in order to be accessible in all test cases.
# Input
x = np.random.randn(1, 3)

# Layer 1 
w_1 = np.random.randn(3,4)
b_1 = np.random.randn(1,4)

# Layer 2 
w_out = np.random.randn(4,2)
b_out = np.random.randn(1,2)

# Output labels
y_target = np.random.randn(1, 2)


input_dim = x.shape[1]
hidden_layer_specs = []
hidden_layer_specs.append({'activation': activation.SigmoidActivation(), 'dim': w_1.shape[1]})
output_dim = y_target.shape[1]

fc_net = network.FCNetwork(input_dim, output_dim, hidden_layer_specs)

fc_net.layers[0].setWeights(w_1)
fc_net.layers[0].setBiases(b_1)

fc_net.layers[1].setWeights(w_out)
fc_net.layers[1].setBiases(b_out)

correct_gradients = sup.Variables()


class TestActivationFunctions(unittest.TestCase):
  '''
  Test activation functions for value and gradient.
  '''
  
  def testUnit(self):
    unitActivation = activation.UnitActivation()
    input = np.random.rand(10)
    self.assertTrue(np.all(input == unitActivation.evaluate(input)))
    self.assertTrue(np.all(np.ones_like(input) == unitActivation.derivative(input)))
  
  def testSigmoid(self):
    sigmoidActivation = activation.SigmoidActivation()
    input = np.random.rand(10)
    self.assertTrue(np.all(1 / (1 + np.exp(-input)) == sigmoidActivation.evaluate(input)))
    self.assertTrue(np.all((1 / (1 + np.exp(-input)) * (1 - (1 / (1 + np.exp(-input)))) == sigmoidActivation.derivative(input))))


class TestLossFunction(unittest.TestCase):
  '''
  Test loss functions for value and gradient.
  '''
  
  def testSquaredErrorLoss(self):
    squared_loss_func = loss.SquaredErrorFunction()
    dim = 10
    y_1 = np.zeros([1, dim])
    y_2 = 2 * np.ones([1, dim])
    squared_error = 0.0
    for i in range(dim):
      squared_error += (y_1[0,i] - y_2[0,i])**2
    squared_error *= 0.5
    # Test value
    self.assertEqual(squared_loss_func.evaluate(y_1, y_2), squared_error, 'Squared error evaluation is wrong.')
    
    # Test gradients
    self.assertTrue(np.all(squared_loss_func.derivative(y_1, y_2) == -(y_2 - y_1)), 'Squared error gradients do not match.')
  

class TestNetwork(unittest.TestCase):

  # Was tested above, can be used now
  sigmoid = activation.SigmoidActivation()

  def testLayer(self):
    nn_layer = layer.FCLayer(0, activation.SigmoidActivation(), input_dim, output_dim)
    self.assertEqual(nn_layer.inputDimension(), input_dim, 'Input dimensions do not match.')
    self.assertEqual(nn_layer.outputDimension(), output_dim, 'Output dimensions do not match.')

  def testNetworkConstruction(self):
    
    # Test number of layers 
    self.assertEqual(len(fc_net.layers), 2, 'Number of layers should be 2 (1 hidden, 1 output).')
    
    # Test hidden layer dimensions
    self.assertTrue(np.all(fc_net.layers[0].getWeights() == w_1), 'Layer 1 weight matrix does not match.')
    self.assertTrue(np.all(fc_net.layers[0].getBiases() == b_1), 'Layer 1 bias matrix does not match.')
    
    # Test output layer dimensions 
    self.assertTrue(np.all(fc_net.layers[1].getWeights() == w_out), 'Output layer weight matrix does not match.')
    self.assertTrue(np.all(fc_net.layers[1].getBiases() == b_out), 'Output layer bias matrix does not match.')

  def testEvaluation(self):
    # Manually compute network output
    tmp_1 = np.dot(x, w_1) + b_1
    h_1 = self.sigmoid.evaluate(tmp_1)
    tmp_2 = np.dot(h_1, w_out) + b_out
    network_output = self.sigmoid.evaluate(tmp_2)
    
    self.assertTrue(np.all(fc_net.evaluateLayer(0, x) == x), 'Evaluation of layer 0 (input) failed.')
    self.assertTrue(np.all(fc_net.evaluateLayer(1, x) == h_1), 'Evaluation of hidden layer failed.')
    self.assertTrue(np.all(fc_net.output(x) == network_output), 'Evaluation of output failed.')

  def testGradients(self):
    
    # General computations
    loss_function = loss.SquaredErrorFunction()
    y = fc_net.output(x)
    loss_derivative = loss_function.derivative(y, y_target)
    
    # Manually compute gradients
    h_1 = fc_net.evaluateLayer(1, x)
    output_sigmoid = self.sigmoid.evaluate(np.dot(h_1, w_out) + b_out) 
    output_sigmoid_derivative = output_sigmoid * (1 - output_sigmoid)
    
    ## Output layer
    delta_out = loss_derivative * output_sigmoid_derivative
    L_w_out = np.dot(h_1.T, delta_out)
    L_b_out = delta_out
    
    ## Hidden layer
    delta_1 = np.dot(delta_out, w_out.T) * self.sigmoid.derivative(np.dot(x, w_1) + b_1)
    L_w_1 = np.dot(x.T, delta_1)
    L_b_1 = delta_1
    
    global correct_gradients
    correct_gradients.weights.append(L_w_1)
    correct_gradients.biases.append(L_b_1)
    
    correct_gradients.weights.append(L_w_out)
    correct_gradients.biases.append(L_b_out)
    
    # Network gradients
    gradients = fc_net.gradients(x, loss_function, y_target)
    ## Output layer
    nn_L_w_out = gradients.weights[1]
    nn_L_b_out = gradients.biases[1]
    
    ## Hidden layer
    nn_L_w_1 = gradients.weights[0]
    nn_L_b_1 = gradients.biases[0]
    
    # Output derivatives
    self.assertTrue(np.all(nn_L_w_out.shape == w_out.shape), 'Gradient shapes of output layer do not match.')
    self.assertTrue(np.all(nn_L_b_out.shape == b_out.shape), 'Gradient shapes of output layer do not match.')
    self.assertTrue(np.all(np.isclose(L_w_out, nn_L_w_out, rtol=1.e-8)))
    self.assertTrue(np.all(np.isclose(L_b_out, nn_L_b_out, rtol=1.e-8)))

    # Hidden layer derivatives
    self.assertTrue(np.all(nn_L_w_1.shape == w_1.shape), 'Gradient shapes of hidden layer do not match.')
    self.assertTrue(np.all(nn_L_b_1.shape == b_1.shape), 'Gradient shapes of hidden layer do not match.')
    self.assertTrue(np.all(np.isclose(L_w_1, nn_L_w_1, rtol=1.e-8)))
    self.assertTrue(np.all(np.isclose(L_b_1, nn_L_b_1, rtol=1.e-8)))
    
    
    
    
class TestSupport(unittest.TestCase):
  '''
  Test support tools.
  '''
  
  def testVariables(self):
    var = sup.Variables()
    var.weights.append(np.ones([2,2]))
    var.weights.append(np.ones([2,2]))
    var.biases.append(np.ones([1,2]))
    var.biases.append(np.ones([1,2]))
    
    # Multiplication 
    var_neg = var * (-1)
    for i in range(len(var)):
      self.assertTrue(np.all(var.weights[i] + var_neg.weights[i] == np.zeros_like(var.weights[i])))
      self.assertTrue(np.all(var.biases[i] + var_neg.biases[i] == np.zeros_like(var.biases[i])))
      
    # Addition
    var_add = var + var_neg
    for i in range(len(var_add)):
      self.assertTrue(np.all(var_add.weights[i] == np.zeros_like(var.weights[i])))
      self.assertTrue(np.all(var_add.biases[i] == np.zeros_like(var.biases[i])))
      
    # Subtraction
    var_sub = var - var
    for i in range(len(var_sub)):
      self.assertTrue(np.all(var_sub.weights[i] == np.zeros_like(var.weights[i])))
      self.assertTrue(np.all(var_sub.biases[i] == np.zeros_like(var.biases[i])))

    # Equality
    self.assertTrue(var == var)
    self.assertFalse(var == var_sub)
    
    # Inequality
    self.assertFalse(var != var)
    self.assertTrue(var != var_sub)

class TestOptimizer(unittest.TestCase):
  '''
  Test the gradient descent optimizer class and methods.
  '''
 

  optimizer = gdo.GradientDescentOptimizer()
 
  def parameterUpdate(self):
    gradients = correct_gradients
    
    p = fc_net.getParameters()
    new_p = p - gradients * self.optimizer.learning_rate
 
    self.assertTrue(np.all(new_p == self.optimizer.getUpdatedParameters(fc_net, gradients)))

 
  def testBatchGradient(self):
    # Manually build a gradient list
    gradient_list = []
    n_layers = 2
    batch_size = 3
    w = np.ones([3,5])
    b = np.ones([1,5])
    grad = sup.Variables()
    for i in range(n_layers):
      grad.weights.append(w)
      grad.biases.append(b)
     
    for j in range(batch_size):
      gradient_list.append(grad)
      
    # Manually compute batch gradient
    batch_gradient_manual = grad * batch_size
    
    batch_gradient = self.optimizer.computeBatchGradient(gradient_list)
        
    self.assertTrue(batch_gradient_manual == batch_gradient)

if __name__ == '__main__':
  unittest.main()
