#!/usr/bin/env python

# Copyright 2017 Mark Pfeiffer, ASL, ETH Zurich, Switzerland
# Copyright 2017 Fadri Furrer, ASL, ETH Zurich, Switzerland
# Copyright 2017 Renaud Dubé, ASL, ETH Zurich, Switzerland

import numpy as np
import pylab as pl
import pickle as pkl
import argparse
import os

import ModelWrapper as model
import GradientDescentOptimizer as optimizer
import FCNetwork as network
import LossFunction as loss
import ActivationFunction as activation
import Support as sup

# Tabula rasa
pl.close('all')

def parse_args():
  parser = argparse.ArgumentParser(description='Train a simple deep NN')
  parser.add_argument('--verbosity', help='Specify verbosity of output. (Options: error, warn, info, debug).', \
                      type=str, default='info')
  parser.add_argument('--plot', help='Activate plotting.', action='store_true', default=False)
  return parser.parse_args()

args = parse_args()
plotting_active = args.plot

# Load data
if os.path.exists('./data/data_train.pkl'):
  data_train = pkl.load(open('./data/data_train.pkl', 'rb'))
else: 
  raise Exception('Training data file does not exist.')
X = data_train[0]
labels_onehot = data_train[1]
n_features = X.shape[1]
n_classes = labels_onehot.shape[1]

# Model setup
# Parameters 
params = model.Params()
params.training_batch_size = 32
params.max_training_steps = 20000
params.learning_rate = 0.01
params.print_steps = 200

# Network
input_dim = n_features
hidden_layer_specs = []
# TODO: update network, add more layers, ...
hidden_layer_specs.append({'activation': activation.SigmoidActivation(), 'dim': 2})
output_dim = n_classes
fc_net = network.FCNetwork(input_dim, output_dim, hidden_layer_specs, output_activation=activation.SigmoidActivation(), loglevel=args.verbosity)
loss_function = loss.SquaredErrorFunction()
gdo = #TODO: instantiate the gradient descent optimizer

nn_model = model.ModelWrapper(network=fc_net, optimizer=gdo, loss_function=loss_function, 
                              x=X, y_target=labels_onehot, params=params)

# Run training
loss = nn_model.train()
nn_model.network.saveModel('./models', 'net_lastname.pkl')

score_train = sup.computeScore(fc_net, X, labels_onehot)
print('The classification score on the training data is {}.'.format(score_train))

# Plotting training specs 
if plotting_active: 
  pl.figure('Loss evolution')
  ax = pl.gca()
  ax.plot(loss[:,0], loss[:,1], c='b', label='training error')
  ax.set_ylim([-0.2, 1.2])
  ax.set_xlabel('Training step')
  ax.set_ylabel('Training error')
  
  # Visualization of classification results
  mesh_step_size = 0.02
  x1_min, x1_max = X[:, 0].min() - .5, X[:, 0].max() + .5
  x2_min, x2_max = X[:, 1].min() - .5, X[:, 1].max() + .5
  xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, mesh_step_size), np.arange(x2_min, x2_max, mesh_step_size))
  predicted_label = np.zeros(xx1.shape)
  
  for ii in range(xx1.shape[0]):
    for jj in range(xx1.shape[1]):
      x_query = np.array([[xx1[ii,jj], xx2[ii,jj]]])
      y_net = fc_net.output(x_query)
      predicted_label[ii,jj] = np.argmax(y_net)
      
  pl.figure('Classification result')
  ax = pl.subplot(111)
  ax.pcolormesh(xx1, xx2, predicted_label, cmap=pl.cm.Spectral)
  # Plot training data
  ax.scatter(X[:, 0], X[:, 1], c=np.argmax(labels_onehot, 1), cmap=pl.cm.Spectral, s=40)
  ax.set_xlabel('x1')
  ax.set_ylabel('x2')

pl.show(block=True)

