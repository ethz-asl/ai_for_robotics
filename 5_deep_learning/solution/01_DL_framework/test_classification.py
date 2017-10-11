# Copyright 2017 Mark Pfeiffer, ASL, ETH Zurich, Switzerland
# Copyright 2017 Fadri Furrer, ASL, ETH Zurich, Switzerland
# Copyright 2017 Renaud Dubé, ASL, ETH Zurich, Switzerland

import numpy as np
import pylab as pl
import pickle as pkl
import argparse
from sklearn import datasets, linear_model, preprocessing
import Support as sup

import FCNetwork as network


def parse_args():
    parser = argparse.ArgumentParser(description='Train a simple deep NN')
    parser.add_argument(
        '--plot',
        help='Activate plotting.',
        action='store_true',
        default=False)
    return parser.parse_args()


args = parse_args()
test_data = pkl.load(open('./data/data_test.pkl', 'rb'))

fc_net = network.FCNetwork()
fc_net.loadModel('./models/', 'trained.pkl')

X_test = test_data[0]
labels_test = test_data[1]

# Compute scores on training and testing data
score_test = sup.computeScore(fc_net, X_test, labels_test)
print('The classification score on the test data is {}'.format(score_test))

# Visualization of classification results
if args.plot:
    mesh_step_size = 0.05
    x1_min, x1_max = X_test[:, 0].min() - .5, X_test[:, 0].max() + .5
    x2_min, x2_max = X_test[:, 1].min() - .5, X_test[:, 1].max() + .5
    xx1, xx2 = np.meshgrid(
        np.arange(x1_min, x1_max, mesh_step_size),
        np.arange(x2_min, x2_max, mesh_step_size))
    predicted_label = np.zeros(xx1.shape)

    for ii in range(xx1.shape[0]):
        for jj in range(xx1.shape[1]):
            x_query = np.array([[xx1[ii, jj], xx2[ii, jj]]])
            y_net = fc_net.output(x_query)
            predicted_label[ii, jj] = np.argmax(y_net)

    pl.figure('Classification result')
    ax = pl.subplot(111)
    ax.pcolormesh(xx1, xx2, predicted_label, cmap=pl.cm.Spectral)
    # Plot training data
    ax.scatter(
        X_test[:, 0],
        X_test[:, 1],
        c=np.argmax(labels_test, 1),
        cmap=pl.cm.Spectral,
        s=40)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')

pl.show(block=True)
