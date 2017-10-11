#!/usr/bin/env python

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
w_1 = np.random.randn(3, 4)
b_1 = np.random.randn(1, 4)

# Layer 2
w_out = np.random.randn(4, 2)
b_out = np.random.randn(1, 2)

# Output labels
y_target = np.random.randn(1, 2)

input_dim = x.shape[1]
hidden_layer_specs = []
hidden_layer_specs.append({
    'activation': activation.SigmoidActivation(),
    'dim': w_1.shape[1]
})
output_dim = y_target.shape[1]

fc_net = network.FCNetwork(input_dim, output_dim, hidden_layer_specs)

fc_net.layers[0].setWeights(w_1)
fc_net.layers[0].setBiases(b_1)

fc_net.layers[1].setWeights(w_out)
fc_net.layers[1].setBiases(b_out)


class TestNetwork(unittest.TestCase):

    # Was tested above, can be used now
    sigmoid = activation.SigmoidActivation()

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
        delta_1 = np.dot(delta_out, w_out.T) * self.sigmoid.derivative(
            np.dot(x, w_1) + b_1)
        L_w_1 = np.dot(x.T, delta_1)
        L_b_1 = delta_1

        # Network gradients
        gradients = fc_net.gradients(x, loss_function, y_target)
        ## Output layer
        nn_L_w_out = gradients.weights[1]
        nn_L_b_out = gradients.biases[1]

        ## Hidden layer
        nn_L_w_1 = gradients.weights[0]
        nn_L_b_1 = gradients.biases[0]

        # Output derivatives
        self.assertTrue(
            np.all(nn_L_w_out.shape == w_out.shape),
            'Gradient shapes of output layer do not match.')
        self.assertTrue(
            np.all(nn_L_b_out.shape == b_out.shape),
            'Gradient shapes of output layer do not match.')
        self.assertTrue(np.all(np.isclose(L_w_out, nn_L_w_out, rtol=1.e-8)))
        self.assertTrue(np.all(np.isclose(L_b_out, nn_L_b_out, rtol=1.e-8)))

        # Hidden layer derivatives
        self.assertTrue(
            np.all(nn_L_w_1.shape == w_1.shape),
            'Gradient shapes of hidden layer do not match.')
        self.assertTrue(
            np.all(nn_L_b_1.shape == b_1.shape),
            'Gradient shapes of hidden layer do not match.')
        self.assertTrue(np.all(np.isclose(L_w_1, nn_L_w_1, rtol=1.e-8)))
        self.assertTrue(np.all(np.isclose(L_b_1, nn_L_b_1, rtol=1.e-8)))


if __name__ == '__main__':
    unittest.main()
