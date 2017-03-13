#!/usr/bin/env python

import pickle as pkl
import numpy as np
import BayesianFiltering


class ProgrammaticError(Exception):
    """Exception raised when method gets called at a wrong time instance.
  Attributes:
      msg  -- The error message to be displayed.
  """

    def __init__(self, msg):
        self.msg = msg
        print("\033[91mERROR: \x1b[0m {}".format(msg))


data = pkl.load(open("data_1.pkl", 'rb'))
measurements = data[0, :]
motions = data[1, :]

if (len(measurements) != len(motions)):
    raise ProgrammaticError(
        'Size of the measurements and motions needs to be the same!')

result = BayesianFiltering.run_bayes_filter(
    measurements, motions, plot_histogram=True)
pkl.dump(result, open("result_bayes_1.pkl", 'wb'))

data = pkl.load(open("data_2.pkl", 'rb'))
measurements = data[0, :]
motions = data[1, :]

if (len(measurements) != len(motions)):
    raise ProgrammaticError(
        'Size of the measurements and motions needs to be the same!')

result = BayesianFiltering.run_bayes_filter(
    measurements, motions, plot_histogram=True)
pkl.dump(result, open("result_bayes_2.pkl", 'wb'))
