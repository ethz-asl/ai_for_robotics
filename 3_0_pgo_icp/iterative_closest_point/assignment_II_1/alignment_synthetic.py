#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 2 10:00 2017
@author: Timo Hinzmann (hitimo@ethz.ch)
"""

import numpy as np
from matplotlib import pyplot
import pylab
from mpl_toolkits.mplot3d import Axes3D
import copy
from numpy import linalg as LA
from scipy.spatial.distance import cdist

def getNearestNeighbors(source, target):
    # TODO
    return indices, mean_error

def computeBestTransformation(source, target):
    source_bar = #TODO
    target_bar = #TODO
    R_hat = computeBestRotation(source, source_bar, target, target_bar)
    t_hat = computeBestTranslation(source_bar, target_bar, R_hat)
    return getTransformationMatrix(R_hat, t_hat)

def getTransformationMatrix(R, t):
    T = np.eye(4)
    # TODO
    return T

def computeBestTranslation(source_bar, target_bar, R):
    # TODO
    return t_opt

def computeBestRotation(source, source_bar, target, target_bar):
    # TODO
    R = np.eye(3)
    return R

def main():
    source_original = np.genfromtxt(open("synthetic_source.xyz"))
    target_original = np.genfromtxt(open("synthetic_target.xyz"))
    source = np.ones((4, source_original.shape[0]))
    target = np.ones((4, target_original.shape[0]))
    source[0:3,:] = np.copy(source_original.T)
    target[0:3,:] = np.copy(target_original.T)

    # Plotting.
    fig = pylab.figure()
    ax = Axes3D(fig)
    ax.scatter(source[0,:], source[1,:], source[2,:], color='red')
    ax.scatter(target[0,:], target[1,:], target[2,:], color='green')
    ax.view_init(azim=69, elev=-97)
    # pyplot.show(block=True)

    # Initialize.
    iter_max = 1000
    convergence_tolerance = 1.0e-16
    previous_mean_error = 1.0e12
    for iter in range(0, iter_max):

        # Get correspondences.
        target_indices, current_mean_error = getNearestNeighbors(source[0:3,:].T, target[0:3,:].T)

        # Compute best transformation.
        T = computeBestTransformation(source[0:3,:].T,target[0:3,target_indices].T)

        # Transform the source pointcloud.
        # TODO

        # Check convergence.
        if abs(previous_mean_error - current_mean_error) < convergence_tolerance:
            print "Converged at iteration: ", iter
            break
        else:
            previous_mean_error = current_mean_error

        # Plotting.
        pyplot.cla()
        ax.scatter(source[0,:], source[1,:], source[2,:], color='red')
        ax.scatter(target[0,:], target[1,:], target[2,:], color='green')
        pyplot.draw()
        ax.view_init(azim=69, elev=-97)
        pyplot.show(block=False)

    # Compute final transformation.
    # TODO
    T_final = np.eye(4)

    print "Final transformation: ", T_final
    np.savetxt('results_alignment_synthetic.txt', T_final)

if __name__ == "__main__":
    main()
