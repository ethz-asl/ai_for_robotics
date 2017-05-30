# A good introduction to TensorFlow layers and CNN can be found here: https://www.tensorflow.org/tutorials/layers
# This exercise has been inspired by Magnus Erik Hvass Pedersen's tutorial on CNN: https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/02_Convolutional_Neural_Network.ipynb

import argparse
import matplotlib.pyplot as plt
import numpy as np
import pickle
import tensorflow as tf
import time

from datetime import timedelta
from IPython import embed 
from tf_helpers import build_conv2d_layer
from tf_helpers import build_fc_layer

DATASET_PATH = "/tmp/deers_and_trucks"
TEST_DATASET_PATH = "/tmp/deers_and_trucks_test"

def load_dataset(path):
  with open(path, "rb") as input_file:
    data = pickle.load(input_file)
  return data['images'], data['cls']

def plot_images(images, cls_names):

    assert len(images) == len(cls_names) == 9

    # Create figure with sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i], cmap = plt.get_cmap('gray'),
                  interpolation='spline16')
            
        # Name of the true class.
        xlabel = cls_names[i]

        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()

# Load the dataset.
images_train, cls_train = load_dataset(DATASET_PATH)
images_test, cls_test = load_dataset(TEST_DATASET_PATH)
img_width = 32
img_height = 32
n_channels = 1
n_classes = 2
cls_names = ["deers", "trucks"]

# Plot a few samples if not disabled.
parser = argparse.ArgumentParser()
parser.add_argument("--disable_visualization", help="disable image visualization",
                    action="store_true")
args = parser.parse_args()
if not(args.disable_visualization):
    plot_images(images_train[0:9], np.asarray(cls_names)[cls_train[0:9]])

# Encode the labels as one hot.
cls_train_one_hot_encoded = np.eye(n_classes, dtype=float)[cls_train]
cls_test_one_hot_encoded = np.eye(n_classes, dtype=float)[cls_test]

# Parameters for first convolutional layer.
conv1_filter_size = 5
conv1_n_filters = 32

# Parameters for second convolutional layer.
conv2_filter_size = 5
conv2_n_filters = 32

# Parameters for fully connected layers.
fc_size1 = 256
fc_size2 = 128

# Create a TensorFlow place holder for the input variables.
x = tf.placeholder(tf.float32, shape=[None, img_width * img_height], name='x')

# Create a TensorFlow place holder for the output variables encoded in one hot format.
y_true = tf.placeholder(tf.float32, shape=[None, n_classes], name='y_true')

# Add a tensor which calculates the true class using argmax.
y_true_cls = tf.argmax(y_true, dimension=1)

# Reshape the input in a format expected by the convolutional layers. 
# -1 signifies that the first dimension will automatically be adjusted to the number of images, i.e. the batch size.
x_image = tf.reshape(x, [-1, img_width, img_height, n_channels])

# First convolutional layer.
first_conv2d_layer =  build_conv2d_layer(x_image,
		                         conv1_filter_size,
		                         conv1_n_filters,
		                         '_conv1')

first_conv2d_layer = tf.nn.max_pool(first_conv2d_layer,
                                    [1, 2, 2, 1],
                                    [1, 2, 2, 1],
                                    'SAME')

first_conv2d_layer = tf.nn.relu(first_conv2d_layer)

# Second convolutional layer.
second_conv2d_layer =  build_conv2d_layer(first_conv2d_layer,
		                         conv2_filter_size, 
		                         conv2_n_filters,
		                         '_conv2')

second_conv2d_layer = tf.nn.max_pool(second_conv2d_layer,
                                    [1, 2, 2, 1],
                                    [1, 2, 2, 1],
                                    'SAME')

second_conv2d_layer = tf.nn.relu(second_conv2d_layer)

# Flatten the output of the second convolutional layer.
layer_flat = tf.contrib.layers.flatten(second_conv2d_layer)

# First fully connected layer.
first_fc_layer = build_fc_layer(layer_flat, fc_size1, '_fc1')
first_fc_layer = tf.nn.relu(first_fc_layer)

# Second fully connected layer.
second_fc_layer = build_fc_layer(first_fc_layer, fc_size2, '_fc2')
second_fc_layer = tf.nn.relu(second_fc_layer)

# Output layer.
output_layer = build_fc_layer(second_fc_layer, n_classes, '_output')

# Prediction layer.
y_pred = tf.nn.softmax(output_layer)
y_pred = tf.argmax(y_pred, dimension=1)

# Create a cost function based on cross entropy. 
# Note that the function calculates the softmax internally so we must use the output of `layer_fc2` directly rather than `y_pred` which has already had the softmax applied.
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_layer, labels=y_true))

reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
cost = tf.add(cost, tf.add_n(reg_losses))

# Create an optimizer.
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

# Create a tensor for computing the accuracy of a network.
accuracy = tf.reduce_mean(tf.cast(tf.equal(y_pred, y_true_cls), tf.float32))

# Create a TensorFlow session and initialize variables.
session = tf.Session()
session.run(tf.global_variables_initializer())

# Create an object responsible of generating random batches.
from batchmaker import Batchmaker
train_batch_size = 64
batchmaker = Batchmaker(images_train, cls_train_one_hot_encoded, train_batch_size)

def make_dictionary(input_data, input_size, output_data, output_size):
  input_values  =  np.zeros([len(input_data)] + input_size)
  output_values  =  np.zeros([len(output_data)] + output_size)
  i = 0
  for input_sample, output_sample in zip(input_data, output_data):
    input_values[i] = np.reshape(input_sample, input_size)
    output_values[i] = np.reshape(output_sample, output_size)
    i += 1
  return {x: input_values, y_true: output_values}

training_dict = make_dictionary(images_train, [img_width * img_height], cls_train_one_hot_encoded, [n_classes])
testing_dict = make_dictionary(images_test, [img_width * img_height], cls_test_one_hot_encoded, [n_classes])

# Function for running the optimizer on random training batches.
def optimize(num_iterations):
    start_time = time.time()

    for i in range(num_iterations):
        # Get a batch of training examples and feed it to the network for optimization.
        x_batch, y_true_batch = batchmaker.next_batch()
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}
        session.run(optimizer, feed_dict=feed_dict_train)

        # Print status every 100 iterations.
        if i % 100 == 0:
            # Calculate and print accuracies.
            acc = session.run(accuracy, feed_dict=feed_dict_train)
            acc_full_train = session.run(accuracy, feed_dict=training_dict)
            acc_full_test = session.run(accuracy, feed_dict=testing_dict)

            msg = "Optimization Iteration: {0:>6}, Training Accuracy On Batch: {1:>6.1%}, Training Accuracy On Full Training Data: {2:>6.1%} , Training Accuracy On Full Testing Data: {3:>6.1%}"
            print(msg.format(i + 1, acc, acc_full_train, acc_full_test))

    end_time = time.time()
    time_dif = end_time - start_time
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))

optimize(num_iterations=1000000)
