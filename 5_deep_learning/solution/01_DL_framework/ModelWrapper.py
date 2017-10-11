import numpy as np
import FCLayer as fc_layer
import pickle as pkl
import os


class Params():

    training_batch_size = 10
    max_training_steps = 1000
    print_steps = 100


class ModelWrapper():

    network = None
    optimizer = None
    loss_function = None
    x = None
    y_target = None
    params = None
    current_idx = 0

    def __init__(self, network, loss_function, optimizer, x, y_target, params):
        self.params = params
        self.network = network
        self.optimizer = optimizer
        self.loss_function = loss_function
        # Training data
        self.x = x
        self.y_target = y_target

    def setNetwork(self, network):
        self.network = network

    def getNetwork(self):
        return self.network

    def setOptimizer(self, optimizer):
        self.optimizer = optimizer

    def getOptimizer(self):
        return self.optimizer

    def getNextDataPoint(self):
        if self.current_idx >= self.x.shape[0]:
            self.current_idx = 0
        input_batch = np.array([self.x[self.current_idx, :]])
        label_batch = np.array([self.y_target[self.current_idx, :]])
        self.current_idx += 1
        return input_batch, label_batch

    def getNextTrainingBatch(self, batch_size=1):
        input_batch = np.zeros([batch_size, self.x.shape[1]])
        label_batch = np.zeros([batch_size, self.y_target.shape[1]])
        for i in range(batch_size):
            X, y_target = self.getNextDataPoint()
            input_batch[i, :] = X
            label_batch[i, :] = y_target
        return input_batch, label_batch

    def prediction(self, x):
        y_net = self.network.output(x)
        y = np.zeros(y_net.shape)
        y[0, np.argmax(y_net)] = 1
        return y

    def train(self):
        step = 0
        loss_evolution = np.zeros([self.params.max_training_steps, 2])
        while step < self.params.max_training_steps:
            x_batch, y_target_batch = self.getNextTrainingBatch(
                self.params.training_batch_size)
            loss = self.optimizer.updateStep(self.network, self.loss_function,
                                             x_batch, y_target_batch)
            loss_evolution[step, :] = np.array([step, loss])
            if step % self.params.print_steps == 0:
                print("Loss in step {} is {}.".format(step, loss))
            step += 1

        return loss_evolution

    def eval(self, x, y_target):
        eval_size = x.shape[0]
        correct_classifications = 0
        for i in range(eval_size):
            prediction = self.prediction(np.array([x[i, :]]))
            truth = np.array([y_target[i, :]])
            if np.all(prediction == truth):
                correct_classifications += 1.0
        return correct_classifications / float(eval_size)

    def loss(self, x, y_target):
        return self.loss_function.evaluate(self.prediction(x), y_target)

    def gradients(self, x, y_target):
        return self.network.gradients(x, self.loss_function, y_target)
