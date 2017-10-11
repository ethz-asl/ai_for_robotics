####################################
# Author: Renaud Dube              #
# Date created: 30.05.2017         #
#                                  #
# Date last changed: 11.10.2017    #
# Changed by: Mark Pfeiffer        #
####################################

import argparse
import numpy as np
import time

from datetime import timedelta

from cnn_model import CNNModel
from utilities import load_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--use_evaluation_dataset", help="use evaluation dataset",
                    action="store_true")
args = parser.parse_args()

if args.use_evaluation_dataset:
  test_dataset_path = "/tmp/deers_and_trucks_evaluation"
else:
  test_dataset_path = "data/deers_and_trucks_test"

# Load the dataset.
images_test, cls_test = load_dataset(test_dataset_path)
n_classes = 2
cls_names = ["deers", "trucks"]

# Encode the labels as one hot.
cls_test_one_hot_encoded = np.eye(n_classes, dtype=float)[cls_test]

# Create a convolutional neural network.
model = CNNModel()

# Load the saved model.
model.load("model/")

# Create a dictionary for evaluating the network on the full validation data.
testing_dict = model.make_dictionary(images_test, cls_test_one_hot_encoded)

# Evaluate and print the accuracy of the network.
acc_full_test = model.sess.run(model.accuracy, feed_dict=testing_dict)
msg = "Training Accuracy On Full Testing Data: {0:>6.1%}"
print(msg.format(acc_full_test))
