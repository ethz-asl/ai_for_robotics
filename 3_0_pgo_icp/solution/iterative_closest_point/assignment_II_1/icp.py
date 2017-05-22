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
    # Uncomment to generate synthetic datasets.
    # Load the input pointclouds.
    #pcl_a = np.genfromtxt(open("pcl_a_bunny.xyz"))
    #pcl_b = copy.deepcopy(pcl_a)

    #T = [[0.9800,0.0098,-0.1987,0.1],
    #     [0.0099,0.9952,0.0978,0.3],
    #     [0.1987,-0.0979,0.9752,0.1],
    #     [0.0, 0.0, 0.0, 1.0]]

    #for i in range(0, pcl_a.shape[0]):
    #  p_hom_a = [pcl_a[i,0], pcl_a[i,1], pcl_a[i,2], 1.0]
    #  p_hom_b = np.dot(T, p_hom_a)
    #  pcl_b[i,:] = [p_hom_b[0], p_hom_b[1], p_hom_b[2]]

    #np.savetxt('synthetic_source.xyz', pcl_a, delimiter=' ')
    #np.savetxt('synthetic_target.xyz', pcl_b, delimiter=' ')

    pcl_a = np.genfromtxt(open("synthetic_source.xyz"))
    pcl_b = np.genfromtxt(open("synthetic_target.xyz"))

    A = pcl_a
    B = pcl_b
    source = np.ones((4,A.shape[0]))
    target = np.ones((4,B.shape[0]))
    source[0:3,:] = np.copy(A.T)
    target[0:3,:] = np.copy(B.T)

    # Plotting.
    fig = pylab.figure()
    ax = Axes3D(fig)
    ax.scatter(source[0,:], source[1,:], source[2,:],color='red')
    ax.scatter(target[0,:], target[1,:], target[2,:],color='green')
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
        source = np.dot(T, source)

        # Check convergence.
        if abs(previous_mean_error - current_mean_error) < convergence_tolerance:
            print "Converged at iteration: ", iter
            break
        else:
            previous_mean_error = current_mean_error

        # Plotting
        pyplot.cla()
        ax.scatter(source[0,:], source[1,:], source[2,:],color='red')
        ax.scatter(target[0,:], target[1,:], target[2,:],color='green')
        pyplot.draw()
        ax.view_init(azim=69, elev=-97)
        pyplot.show(block=False)

    T_final = computeBestTransformation(A, source[0:3,:].T)
    print "Final transformation: ", T_final
    np.savetxt('results_synthetic.txt', T_final)

if __name__ == "__main__":
    main()
