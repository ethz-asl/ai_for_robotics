#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 2 10:00 2017
@author: Timo Hinzmann (hitimo@ethz.ch)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import linalg as sla
from enum import Enum
import copy
import pylab

import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix

def plot_coo_matrix(m):
    if not isinstance(m, coo_matrix):
        m = coo_matrix(m)
    fig = plt.figure()
    ax = fig.add_subplot(111, axisbg='black')
    ax.plot(m.col, m.row, 's', color='white', ms=1)
    ax.set_xlim(0, m.shape[1])
    ax.set_ylim(0, m.shape[0])
    ax.set_aspect('equal')
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    return ax

# Types of measurements
class ConstraintType(Enum):
    ODOMETRY = 0
    LOOP_CLOSURE = 1
    FIXED_NODE = 2

class PoseGraphOptimization1D():
    def __init__(self, num_nodes):
        # Position estimates
        self.x = np.zeros([num_nodes, 1])
        self.constraints = []
        self.num_nodes = num_nodes
        self.dimensions = 1

    #    i:  index of the i'th node
    #    j:  index of the j'th node
    # u_ij:  odometry measurement from node i to node j
    def handleOdometryMeasurement(self, i, j, u_ij):
        self.integrateOdometryMeasurement(i, j, u_ij)
        self.addOdometryConstraintToPosegraph(i, j, u_ij)

    #    i:  index of the i'th node
    #    j:  index of the j'th node
    # u_ij:  odometry measurement from node i to node j
    def integrateOdometryMeasurement(self, i, j, u_ij):
        self.x[j] = self.x[i] + u_ij

    #    i:  index of the i'th node
    #    j:  index of the j'th node
    # u_ij:  odometry measurement from node i to node j
    def addOdometryConstraintToPosegraph(self, i, j, u_ij):
        # Jacobian wrt. pose/node i
        A_ij = 1.0
        # Jacobina wrt. pose/node j
        B_ij = -1.0
        # Information of the measurement
        Omega_ij = 100 #self.Omega[i,i]
        # Add the constraint to the pose graph
        self.constraints.append([ConstraintType.ODOMETRY, i, j, A_ij, B_ij, Omega_ij, u_ij])

    def handleLoopClosure(self, i, j):
        # Jacobian wrt. pose/node i
        A_ij = 1.0
        # Jacobian wrt. pose/node j
        B_ij = -1.0
        # Information of the measurement
        Omega_ij = 100 #self.Omega[i,j]
        u_ij = 0.0
        # Add the constraint to the pose graph
        self.constraints.append([ConstraintType.LOOP_CLOSURE, i, j, A_ij, B_ij, Omega_ij, u_ij])

    def fixFirstNode(self):
        # Jacobian wrt. pose/node i
        A_ij = 1.0
        B_ij = 0.0
        # Information of the measurement
        Omega_ij = 1000
        u_ij = 0.0
        i = j = 0
        # Add the constraint to the pose graph
        self.constraints.append([ConstraintType.FIXED_NODE, i, j, A_ij, B_ij, Omega_ij, u_ij])

    def evaluateOdometryResidual(self, i, j, u_ij):
        return self.x[i] - self.x[j] + u_ij

    def evaluateLoopClosureResidual(self, i, j):
        return self.x[i] - self.x[j]

    def evaluateFixedNodeResidual(self):
        return 0.0

    def optimizePoseGraph(self):
        num_optimization_iterations = 2
        num_constraints = len(self.constraints)
        x_odo = copy.deepcopy(self.x)
        for iter in range(0, num_optimization_iterations):
            H = np.zeros([self.num_nodes, self.num_nodes])
            b = np.zeros([self.num_nodes, 1])
            for constraint in self.constraints:
                constraint_type = constraint[0]
                i = constraint[1]
                j = constraint[2]
                A_ij = constraint[3]
                B_ij = constraint[4]
                Omega_ij = constraint[5]
                u_ij = constraint[6]
                # Evaluate the residuals
                e_ij = 0.0
                if constraint_type == ConstraintType.ODOMETRY:
                    e_ij = self.evaluateOdometryResidual(i, j, u_ij)

                if constraint_type == ConstraintType.LOOP_CLOSURE:
                    e_ij = self.evaluateLoopClosureResidual(i, j)

                if constraint_type == ConstraintType.FIXED_NODE:
                    e_ij = self.evaluateFixedNodeResidual()

                b[i] += np.dot(A_ij, np.dot(Omega_ij, e_ij))
                b[j] += np.dot(B_ij, np.dot(Omega_ij, e_ij))

                # Hessian
                H[i,i] += np.dot(A_ij, np.dot(Omega_ij, A_ij))
                H[i,j] += np.dot(A_ij, np.dot(Omega_ij, B_ij))
                H[j,i] += np.dot(B_ij, np.dot(Omega_ij, A_ij))
                H[j,j] += np.dot(B_ij, np.dot(Omega_ij, B_ij))

            # Solve the linear system.
            delta_x = sla.spsolve(H, -b)
            delta_x = delta_x.reshape(self.num_nodes, self.dimensions)

            # Update the states by applying the increments.
            self.x += delta_x

        k = np.arange(0, self.num_nodes, 1)
        gt = np.array([0.0,1.0,2.0,3.0,0.0])
        pylab.plot(k, gt, 'k', label ='Ground truth')
        pylab.plot(k, x_odo, 'r', label='Odometry integrated')
        pylab.plot(k, self.x, 'g', label='Estimates')
        pylab.title('Position Estimates')
        pylab.xlabel('Timestep k [-]')
        pylab.ylabel('Position x [m]')
        pylab.legend(loc='upper left')
        np.savetxt('results_1D.txt', self.x)

        print np.nonzero(abs(H))
        B = abs(H) > 0
        B.astype(np.int)
        plt.matshow(B, cmap=plt.cm.binary)
        plt.title('Hessian')
        plt.show()


def main():
    # Relative path to data from exercise sheet.
    base = "../../../pose_graph_optimization/assignment_I_1/"

    input_data = np.genfromtxt(open(base + "data_1D.txt"))
    num_measurements = input_data.shape[0]
    print "Number of rows: ", num_measurements

    num_nodes = 5
    pgo = PoseGraphOptimization1D(num_nodes)
    # Keep the first node fixed at origin.
    pgo.fixFirstNode()
    for iter in range(0, num_measurements):
        print "Iter ", iter, "/", num_measurements
        if input_data[iter, 0] == 0:
            i = int(input_data[iter, 1])
            j = int(input_data[iter, 2])
            u_ij = input_data[iter, 3]
            print("Received an odometry measurement from node_%1d to node_%1d with meas. distance of %1f" % (i, j, u_ij))
            pgo.handleOdometryMeasurement(i,j,u_ij)

        if input_data[iter, 0] == 1:
            print("Found a Loop-Closure from node_%1d to node_%1d!" % (i, j))
            i = int(input_data[iter, 1])
            j = int(input_data[iter, 2])
            pgo.handleLoopClosure(i,j)

    pgo.optimizePoseGraph()

if __name__ == "__main__":
    main()
