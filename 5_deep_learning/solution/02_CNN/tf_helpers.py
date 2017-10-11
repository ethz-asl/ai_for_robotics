# Copyright 2017 Mark Pfeiffer, ASL, ETH Zurich, Switzerland
# Copyright 2017 Fadri Furrer, ASL, ETH Zurich, Switzerland
# Copyright 2017 Renaud Dubé, ASL, ETH Zurich, Switzerland

# This file defines a few helper functions to build and optimize tensorflow graphs.
import numpy as np
import tensorflow as tf
from IPython import embed


def build_conv2d_layer(previous_layer,
                       filter_size,
                       n_filters,
                       var_suffix,
                       stride_length=1,
                       regularizer_weight=0.001):

    # The number of channels is the last element of the shape.
    n_channels_previous_layer = previous_layer.get_shape()[-1]

    weights = tf.get_variable(
        "weights" + var_suffix,
        [filter_size, filter_size, n_channels_previous_layer,
         n_filters], tf.float32,
        tf.contrib.layers.xavier_initializer(),
        tf.contrib.layers.l2_regularizer(regularizer_weight))

    biases = tf.get_variable("biases" + var_suffix, n_filters, tf.float32,
                             tf.constant_initializer(0))

    conv_layer = tf.nn.conv2d(previous_layer, weights,
                              [1, stride_length, stride_length, 1], 'SAME')

    conv_layer = tf.nn.bias_add(conv_layer, biases)

    return conv_layer


def build_fc_layer(previous_layer,
                   n_neurons,
                   var_suffix,
                   regularizer_weight=0.001):

    # The size is the last element of the shape.
    previous_layer_size = previous_layer.get_shape()[-1]

    weights = tf.get_variable(
        "weights" + var_suffix, [previous_layer_size, n_neurons], tf.float32,
        tf.contrib.layers.xavier_initializer(),
        tf.contrib.layers.l2_regularizer(regularizer_weight))

    biases = tf.get_variable("biases" + var_suffix, n_neurons, tf.float32,
                             tf.constant_initializer(0))

    return tf.matmul(previous_layer, weights) + biases
