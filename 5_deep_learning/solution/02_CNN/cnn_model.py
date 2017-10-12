# Copyright 2017 Mark Pfeiffer, ASL, ETH Zurich, Switzerland
# Copyright 2017 Fadri Furrer, ASL, ETH Zurich, Switzerland
# Copyright 2017 Renaud Dubé, ASL, ETH Zurich, Switzerland

import tensorflow as tf
import numpy as np
import os

from tf_helpers import build_conv2d_layer
from tf_helpers import build_fc_layer


class CNNModel():
    def __init__(self):

        # Dataset specific parameters.
        self.img_width = 32
        self.img_height = 32
        self.n_channels = 1
        self.n_classes = 2

        # Parameters for first convolutional layer.
        self.conv1_filter_size = 5
        self.conv1_n_filters = 16

        # Parameters for second convolutional layer.
        self.conv2_filter_size = 5
        self.conv2_n_filters = 16

        # Parameters for fully connected layers.
        self.fc_size1 = 256
        self.fc_size2 = 128

        # Create a TensorFlow session and initialize variables.
        tf.reset_default_graph()
        self.sess = tf.Session()

        # Create a TensorFlow place holder for the input variables.
        self.x = tf.placeholder(
            tf.float32,
            shape=[None, self.img_width * self.img_height],
            name='x')

        # Create a TensorFlow place holder for the output variables encoded in one hot format.
        self.y_true = tf.placeholder(
            tf.float32, shape=[None, self.n_classes], name='y_true')

        # Add a tensor which calculates the true class using argmax.
        self.y_true_cls = tf.argmax(self.y_true, dimension=1)

        # Reshape the input in a format expected by the convolutional layers.
        # -1 signifies that the first dimension will automatically be adjusted to the number of images, i.e. the batch size.
        self.x_image = tf.reshape(
            self.x, [-1, self.img_width, self.img_height, self.n_channels])

        # First convolutional layer.
        self.first_conv2d_layer = build_conv2d_layer(
            self.x_image, self.conv1_filter_size, self.conv1_n_filters,
            '_conv1')

        self.first_conv2d_layer = tf.nn.max_pool(
            self.first_conv2d_layer, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

        self.first_conv2d_layer = tf.nn.relu(self.first_conv2d_layer)

        # Second convolutional layer.
        self.second_conv2d_layer = build_conv2d_layer(
            self.first_conv2d_layer, self.conv2_filter_size,
            self.conv2_n_filters, '_conv2')

        self.second_conv2d_layer = tf.nn.max_pool(
            self.second_conv2d_layer, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

        self.second_conv2d_layer = tf.nn.relu(self.second_conv2d_layer)

        # Flatten the output of the second convolutional layer.
        self.layer_flat = tf.contrib.layers.flatten(self.second_conv2d_layer)

        # First fully connected layer.
        self.first_fc_layer = build_fc_layer(self.layer_flat, self.fc_size1,
                                             '_fc1')
        self.first_fc_layer = tf.nn.relu(self.first_fc_layer)

        # Second fully connected layer.
        self.second_fc_layer = build_fc_layer(self.first_fc_layer,
                                              self.fc_size2, '_fc2')
        self.second_fc_layer = tf.nn.relu(self.second_fc_layer)

        # Output layer.
        self.output_layer = build_fc_layer(self.second_fc_layer,
                                           self.n_classes, '_output')

        # Prediction layer.
        self.y_pred = tf.nn.softmax(self.output_layer)
        self.y_pred = tf.argmax(self.y_pred, dimension=1)

        # Create a cost function based on cross entropy.
        # Note that the function calculates the softmax internally so we must use the output of `layer_fc2`
        # directly rather than `y_pred` which has already had the softmax applied.
        self.cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=self.output_layer, labels=self.y_true))

        self.reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.cost = tf.add(self.cost, tf.add_n(self.reg_losses))

        # Create an optimizer.
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=1e-4).minimize(self.cost)

        # Create a tensor for computing the accuracy of a network.
        self.accuracy = tf.reduce_mean(
            tf.cast(tf.equal(self.y_pred, self.y_true_cls), tf.float32))

        # Initialize the model's variables.
        self.sess.run(tf.global_variables_initializer())

        # Create an object for saving and loading a model.
        self.saver = tf.train.Saver()

    def make_dictionary(self, input_data, output_data):
        input_values = np.zeros([len(input_data)] +
                                [self.img_width * self.img_height])
        output_values = np.zeros([len(output_data)] + [self.n_classes])
        i = 0
        for input_sample, output_sample in zip(input_data, output_data):
            input_values[i] = np.reshape(input_sample,
                                         [self.img_width * self.img_height])
            output_values[i] = np.reshape(output_sample, [self.n_classes])
            i += 1
        return {self.x: input_values, self.y_true: output_values}

    def save(self, model_path):
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        self.saver.save(self.sess, model_path + '/model-final.ckpt')

    def load(self, model_path):
        print('Loading session from "{}"'.format(model_path))
        ckpt = tf.train.get_checkpoint_state(model_path)
        print('Restoring model {}'.format(ckpt.model_checkpoint_path))
        self.saver.restore(self.sess, ckpt.model_checkpoint_path)
