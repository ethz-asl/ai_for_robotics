#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 2 10:00 2017
@author: Timo Hinzmann (hitimo@ethz.ch)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import linalg as sla
from scipy import array, linalg, dot
from enum import Enum
import copy
import pylab

def main():

    vertices = np.genfromtxt(open("vertices.dat"))
    edges = np.genfromtxt(open("edges.dat"))
    lc = np.genfromtxt(open("loop_closures.dat"))

    pylab.plot(vertices[:,1], vertices[:,2], 'b')
    plt.pause(5)

    # TODO
    # Create and optimize the pose graph.
    # Feel free to reuse code from the 1D robot exercise.

    # Save the optimized states in rows: [x_0, y_0, th_0; x_1, y_1, th_1; ...]
    x_opt = [vertices[:,1].T, vertices[:,2].T, vertices[:,3].T]
    np.savetxt('results_2D.txt', np.transpose(x_opt))

if __name__ == "__main__":
    main()
