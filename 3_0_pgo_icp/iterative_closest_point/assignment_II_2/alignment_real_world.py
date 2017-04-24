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

def main():
    source_original = np.genfromtxt(open("vision_source.xyz"))
    target_original = np.genfromtxt(open("laser_target.xyz"))
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
    pyplot.show(block=True)

    # Align 'source' to 'target' pointcloud and compute the final transformation.
    # Feel free to reuse code from the previous exercise.
    # TODO
    T_final = np.eye(4)

    # Don't forget to save the final transformation.
    print "Final transformation: ", T_final
    np.savetxt('results_alignment_real_world.txt', T_final)

if __name__ == "__main__":
    main()
