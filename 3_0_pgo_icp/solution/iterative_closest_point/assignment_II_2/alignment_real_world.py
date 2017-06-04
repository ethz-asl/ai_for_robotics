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
import matplotlib.pyplot as plt, numpy as np, numpy.random, scipy

def getNearestNeighbors(source, target):
    euclidean_distances = cdist(source, target, 'euclidean')
    indices = euclidean_distances.argmin(axis=1)
    errors = euclidean_distances[np.arange(euclidean_distances.shape[0]), indices]
    mean_error = np.sum(errors) / errors.size
    return indices, mean_error

def computeBestTransformation(source, target):
    N = source.shape[0]
    sum_source = 0.0
    sum_target = 0.0
    for i in range(0, source.shape[0]):
        sum_source += source[i]
        sum_target += target[i]
    source_bar = np.dot(1.0/N, sum_source)
    target_bar = np.dot(1.0/N, sum_target)
    R_hat = computeBestRotation(source, source_bar, target, target_bar)
    t_hat = computeBestTranslation(source_bar, target_bar, R_hat)
    return getTransformationMatrix(R_hat, t_hat)

def getTransformationMatrix(R,t):
    T = np.eye(4)
    T[0:3,0:3] = R
    T[0:3,3] = t
    return T

def computeBestTranslation(source_bar, target_bar, R):
    t_opt = target_bar.transpose() - np.dot(R, source_bar.transpose())
    return t_opt

def computeBestRotation(source, source_bar, target, target_bar):
    M = np.zeros(shape=(3,3))
    for i in range(0, source.shape[0]):
        M += np.outer(source[i,:]-source_bar,np.transpose(target[i,:]-target_bar))
    U, D, V = np.linalg.svd(M)
    if  np.linalg.det(U)*np.linalg.det(V) < 0:
      for x in range (0,3):
          V[2, x] *= -1;

    R = np.dot(V.T, U.T)
    return R

def main():
    # Relative path to data from exercise sheet.
    base = "../../../iterative_closest_point/assignment_II_2/"

    source_original = np.genfromtxt(open(base + "vision_source.xyz"))
    target_original = np.genfromtxt(open(base + "laser_target.xyz"))
    source = np.ones((4, source_original.shape[0]))
    target = np.ones((4, target_original.shape[0]))
    source[0:3,:] = np.copy(source_original[:,0:3].T)
    target[0:3,:] = np.copy(target_original[:,0:3].T)

    # Plotting.
    fig = pylab.figure()
    ax = Axes3D(fig)
    # Visualize only every 10th point. You can change this for quicker visualization.
    n = 10
    source_vis = np.copy(source[:,::n])
    target_vis = np.copy(target[:,::n])
    ax.scatter(source_vis[0,:], source_vis[1,:], source_vis[2,:], color='red', lw = 0, s=1)
    ax.scatter(target_vis[0,:], target_vis[1,:], target_vis[2,:], color='green', lw = 0, s=1)
    # Make sure that the aspect ratio is equal for x/y/z axis.
    X = target_vis[0,:]
    Y = target_vis[1,:]
    Z = target_vis[2,:]
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0
    mid_x = (X.max()+X.min()) * 0.5
    mid_y = (Y.max()+Y.min()) * 0.5
    mid_z = (Z.max()+Z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    ax.set_aspect('equal')
    #pyplot.show(1)

    # Align 'source' to 'target' pointcloud and compute the final transformation.
    # Initialize.
    iter_max = 1000
    convergence_tolerance = 1.0e-3
    previous_mean_error = 1.0e12
    for iter in range(0, iter_max):

        # Downsampling/Filtering of the source cloud:
        # REMARK: There are many possible approaches to deal with large point-clouds.
        # Usually the space is divided into cells in which every cell only
        # allows a maximum number of points. The nearest neighbor search is
        # then performed using efficient KD-Tree implementations.
        # In this example, we simply select N random points from the source
        # point-cloud, chaning the selection in every iteration.

        # Select 5000 samples from soure cloud (without placing back).
        source_filtered = source.T
        source_filtered = source_filtered[np.random.choice(source_filtered.shape[0], 5000, replace=False)]
        source_filtered = source_filtered.T

        # Get correspondences.
        target_indices, current_mean_error = getNearestNeighbors(source_filtered[0:3,:].T, target[0:3,:].T)

        # Compute best transformation.
        T = computeBestTransformation(source_filtered[0:3,:].T,target[0:3,target_indices].T)

        # Transform the source pointcloud.
        source = np.dot(T, source)

        # Check convergence.
        delta_error = abs(previous_mean_error - current_mean_error)
        if  delta_error < convergence_tolerance:
            print "Converged at iteration: ", iter
            break
        else:
            previous_mean_error = current_mean_error

        print "Iteration: ", iter, "  current mean error: ", current_mean_error, "  delta_error: ", delta_error

    T_final = computeBestTransformation(source_original, source[0:3,:].T)

    # Don't forget to save the final transformation.
    print "Final transformation: ", T_final
    np.savetxt('results_alignment_real_world.txt', T_final)

if __name__ == "__main__":
    main()
