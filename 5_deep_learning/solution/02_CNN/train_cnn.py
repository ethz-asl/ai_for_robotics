####################################
# Author: Renaud Dube              #
# Date created: 30.05.2017         #
#                                  #
# Date last changed: 11.10.2017    #
# Changed by: Renaud Dube          #
####################################


# A good introduction to TensorFlow layers and CNN can be found here: https://www.tensorflow.org/tutorials/layers
# This exercise has been inspired by Magnus Erik Hvass Pedersen's tutorial on CNN: https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/02_Convolutional_Neural_Network.ipynb
import tensorflow as tf
import argparse
import numpy as np
import time

from datetime import timedelta

from batchmaker import Batchmaker
from cnn_model import CNNModel
from utilities import load_dataset, plot_images

DATASET_PATH = "data/deers_and_trucks"

# Load the dataset.
images_train, cls_train = load_dataset(DATASET_PATH)
n_classes = 2
cls_names = ["deers", "trucks"]

# Plot a few samples if not disabled.
parser = argparse.ArgumentParser()
parser.add_argument(
    "--disable_visualization",
    help="disable image visualization",
    action="store_true")
args = parser.parse_args()
if not (args.disable_visualization):
    plot_images(images_train[0:9], np.asarray(cls_names)[cls_train[0:9]])

# Encode the labels as one hot.
cls_train_one_hot_encoded = np.eye(n_classes, dtype=float)[cls_train]

# Create a convolutional neural network.
model = CNNModel()

# Create an object responsible of generating random batches.
train_batch_size = 64
batchmaker = Batchmaker(images_train, cls_train_one_hot_encoded,
                        train_batch_size)

# Create a dictionary for evaluating the network on the full training data.
training_dict = model.make_dictionary(images_train, cls_train_one_hot_encoded)


# Function for running the optimizer on random training batches.
def optimize(num_iterations):
    start_time = time.time()

    for i in range(num_iterations):
        # Get a batch of training examples and feed it to the network for optimization.
        x_batch, y_true_batch = batchmaker.next_batch()
        feed_dict_train = {model.x: x_batch, model.y_true: y_true_batch}
        model.sess.run(model.optimizer, feed_dict=feed_dict_train)

        # Print status every x iterations.
        if i % 5 == 0:
            # Calculate and print accuracies.
            acc = model.sess.run(model.accuracy, feed_dict=feed_dict_train)
            acc_full_train = model.sess.run(
                model.accuracy, feed_dict=training_dict)

            msg = "Optimization Iteration: {0:>6}, Training Accuracy On Batch: {1:>6.1%}, Training Accuracy On Full Training Data: {2:>6.1%}."
            print(msg.format(i + 1, acc, acc_full_train))

    end_time = time.time()
    time_dif = end_time - start_time
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))


optimize(num_iterations=100)

model.save("model/")
