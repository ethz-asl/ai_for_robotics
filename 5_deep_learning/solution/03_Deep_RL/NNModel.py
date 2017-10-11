####################################
# Author: Mark Pfeiffer            #
# Date created: 30.05.2017         #
#                                  #
# Date last changed: 11.10.2017    #
# Changed by: Mark Pfeiffer        #
####################################

import tensorflow as tf
import numpy as np


class NNModel():
    """NN model for reinforcement learning that maps from state to action
    probabilities."""

    def __init__(self, learning_rate=0.01):
        self.input_dim = 4  # physical state of the system [cart position, cart velocity, pole angle, pole rotational velocity]
        self.output_dim = 1  # action to be taken (left or right)
        self.hidden_layer_size = [
            50, 20
        ]  # numbers of hidden units for the 2 hidden layers

        #### NETWORK DEFINITION ####
        self.input = tf.placeholder(
            dtype=tf.float32,
            shape=[None, self.input_dim],
            name='observations')
        # Layer 1
        Wfc1 = tf.get_variable(
            'Wfc1',
            shape=[self.input_dim, self.hidden_layer_size[0]],
            initializer=tf.contrib.layers.xavier_initializer())
        relu1 = tf.nn.relu(tf.matmul(self.input, Wfc1), name='relu1')
        #Layer 2
        Wfc2 = tf.get_variable(
            'Wfc2',
            shape=[self.hidden_layer_size[0], self.hidden_layer_size[1]],
            initializer=tf.contrib.layers.xavier_initializer())
        relu2 = tf.nn.relu(tf.matmul(relu1, Wfc2), name='relu2')
        # Layer 3
        Wfc3 = tf.get_variable(
            'Wfc3',
            shape=[self.hidden_layer_size[1], self.output_dim],
            initializer=tf.contrib.layers.xavier_initializer())
        score = tf.matmul(relu2, Wfc3)

        # probability of taking action 1 (right)
        self.action_probability = tf.nn.sigmoid(score)

        #### LOSS FUNCTION ####
        self.negated_action = tf.placeholder(
            dtype=tf.float32,
            shape=[None, self.output_dim],
            name='action_taken')
        self.reward_signal = tf.placeholder(
            dtype=tf.float32, shape=[None, 1], name='reward_signal')

        # Loss function
        # likelihood of the chosen action given the probability output of the model
        # (negated_action stands for the action which was not taken)
        self.likelihood = self.negated_action * (self.negated_action - self.action_probability) + \
                          (1 - self.negated_action) * (self.negated_action + self.action_probability)
        log_likelihood = tf.log(self.likelihood)
        # overall reward (whole episode) would be the likelihood * reward signal
        #   -> the likelihood of choosing the action which gives high reward has to be maximized
        self.loss = -tf.reduce_mean(log_likelihood * self.reward_signal)
        # gradients = tf.gradients(loss, training_vars)

        #### NETWORK OPTIMIZATION ####
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        # compute (gradient, variable) pairs
        self.gradient_variable_pairs = optimizer.compute_gradients(self.loss)
        self.Wfc1_grad = tf.placeholder(dtype=tf.float32, name='batch_grad1')
        self.Wfc2_grad = tf.placeholder(dtype=tf.float32, name='batch_grad2')
        self.Wfc3_grad = tf.placeholder(dtype=tf.float32, name='batch_grad3')
        batch_grad = [self.Wfc1_grad, self.Wfc2_grad, self.Wfc3_grad]
        self.training_vars = tf.trainable_variables()
        batch_gradient_variable_pairs = zip(batch_grad, self.training_vars)
        self.update_grads = optimizer.apply_gradients(
            batch_gradient_variable_pairs)
