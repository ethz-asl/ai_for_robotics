####################################
# Author: Mark Pfeiffer            #
# Date created: 30.05.2017         #
#                                  #
# Date last changed: 11.10.2017    #
# Changed by: Mark Pfeiffer        #
####################################

from sklearn import datasets
import numpy as np
import pickle as pkl
import os

np.random.seed(100)

n_classes = 6
n_features = 2
n_samples = 10000
ratio_train = 0.8
X, labels = datasets.make_blobs(centers=n_classes, n_features=n_features, n_samples=n_samples, center_box=(-2,2), cluster_std=0.3, shuffle=True)
labels_onehot = np.eye(n_classes, dtype=float)[labels]

cutoff_idx = int(ratio_train * n_samples)
data_train = (X[:cutoff_idx, :], labels_onehot[:cutoff_idx, :])
data_test = (X[cutoff_idx:, :], labels_onehot[cutoff_idx:, :])

directory = 'data'
if not os.path.isdir(directory):
  os.mkdir(directory)

pkl.dump(data_train, open(os.path.join(directory, 'data_train.pkl'), 'wb'))
pkl.dump(data_test, open(os.path.join(directory, 'data_test.pkl'), 'wb'))

print('Successfully generated data and saved to directory "{}".'.format(directory))


