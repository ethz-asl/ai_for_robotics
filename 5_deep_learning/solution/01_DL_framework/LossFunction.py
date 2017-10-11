####################################
# Author: Fadri Furrer             #
# Date created: 30.05.2017         #
#                                  #
# Date last changed: 11.10.2017    #
# Changed by: Mark Pfeiffer        #
####################################

import numpy as np
from abc import ABCMeta, abstractmethod


class LossFunction():
    """Loss function base class."""
    __metaClass__ = ABCMeta

    @abstractmethod
    def evaluate(self, y, y_target):
        pass

    @abstractmethod
    def derivative(self, y, y_target):
        pass


class SquaredErrorFunction(LossFunction):
    """Squared error loss function."""

    def evaluate(self, y, y_target):
        return 0.5 * np.sum(np.dot((y_target - y), (y_target - y).T))

    def derivative(self, y, y_target):
        return -(y_target - y)
