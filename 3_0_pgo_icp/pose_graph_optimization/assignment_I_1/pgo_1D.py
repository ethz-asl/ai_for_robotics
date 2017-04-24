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
        self.x[j] = #TODO

    #    i:  index of the i'th node
    #    j:  index of the j'th node
    # u_ij:  odometry measurement from node i to node j
    def addOdometryConstraintToPosegraph(self, i, j, u_ij):
        # Jacobian wrt. pose/node i.
        A_ij = #TODO
        # Jacobian wrt. pose/node j.
        B_ij = #TODO
        # Information of the measurement
        Omega_ij = #TODO
        # Add the constraint to the pose graph.
        self.constraints.append([ConstraintType.ODOMETRY, i, j, A_ij, B_ij, Omega_ij, u_ij])

    def handleLoopClosure(self, i, j):
        # Jacobian wrt. pose/node i.
        A_ij = #TODO
        # Jacobian wrt. pose/node j.
        B_ij = #TODO
        # Information of the measurement.
        Omega_ij = #TODO
        u_ij = #TODO
        # Add the constraint to the pose graph.
        self.constraints.append([ConstraintType.LOOP_CLOSURE, i, j, A_ij, B_ij, Omega_ij, u_ij])

    def fixFirstNode(self):
        # Jacobian wrt. pose/node i.
        A_ij = #TODO
        B_ij = #TODO
        # Information of the measurement.
        Omega_ij = #TODO
        u_ij = #TODO
        i = #TODO
        j = #TODO
        # Add the constraint to the pose graph.
        self.constraints.append([ConstraintType.FIXED_NODE, i, j, A_ij, B_ij, Omega_ij, u_ij])

    def evaluateOdometryResidual(self, i, j, u_ij):
        e = #TODO
        return e

    def evaluateLoopClosureResidual(self, i, j):
        e = #TODO
        return e

    def evaluateFixedNodeResidual(self):
        e = #TODO
        return e

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

                b[i] = #TODO
                b[j] = #TODO

                # Hessian
                H[i,i] = #TODO
                H[i,j] = #TODO
                H[j,i] = #TODO
                H[j,j] = #TODO

            # Sparse Cholesky Factorization
            self.x = #TODO

        k = np.arange(0, self.num_nodes, 1)
        pylab.plot(k, x_odo, 'r', label='Odometry integrated')
        pylab.plot(k, self.x, 'g', label='Estimates')
        pylab.title('Position Estimates')
        pylab.xlabel('Timestep k [-]')
        pylab.ylabel('Position x [m]')
        pylab.legend(loc='upper left')

        # Save the results.
        np.savetxt('results_1D.txt', self.x)

        print np.nonzero(abs(H))
        B = abs(H) > 0
        B.astype(np.int)
        plt.matshow(B, cmap=plt.cm.binary)
        plt.title('Hessian')
        plt.show()


def main():
    input_data = np.genfromtxt(open("data_1D.txt"))
    num_measurements = input_data.shape[0]
    print "Number of measurements: ", num_measurements

    num_nodes = num_measurements
    pgo = PoseGraphOptimization1D(num_nodes)
    # Keep the first node fixed at origin.
    pgo.fixFirstNode()
    for iter in range(0, num_measurements):
        print "Iter ", iter, "/", num_measurements
        if input_data[iter, 0] == ConstraintType.ODOMETRY.value:
            i = int(input_data[iter, 1])
            j = int(input_data[iter, 2])
            u_ij = input_data[iter, 3]
            print("Received an odometry measurement from node_%1d to node_%1d with meas. distance of %1f" % (i, j, u_ij))
            pgo.handleOdometryMeasurement(i, j, u_ij)

        if input_data[iter, 0] == ConstraintType.LOOP_CLOSURE.value:
            print("Found a Loop-Closure from node_%1d to node_%1d!" % (i, j))
            i = int(input_data[iter, 1])
            j = int(input_data[iter, 2])
            pgo.handleLoopClosure(i,j)

    pgo.optimizePoseGraph()

if __name__ == "__main__":
    main()
