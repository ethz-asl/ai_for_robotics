#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 2 10:00 2017
@author: Timo Hinzmann (hitimo@ethz.ch)
"""
import math
from math import floor, ceil
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import linalg as sla
from scipy import array, linalg, dot
from enum import Enum
import copy
import pylab

# References:
# [1] Grisetti, Kuemmerle, Stachniss et al. "A Tutorial on Graph-Based SLAM"

# Pose-graph optimization closely following Algorithm 1, 2D from [1].
class PoseGraphOptimization2D():
    def __init__(self, vertices, constraints):
        self.vertices = vertices
        self.constraints = constraints

        # State x := [x,y,theta].
        self.x = self.vertices[:, 1:]
        self.index_x = 0
        self.index_y = 1
        self.index_theta = 2

        # Dimensions.
        self.num_nodes = self.vertices.shape[0]
        self.dimensions = 3
        self.num_states = self.dimensions * self.num_nodes

    # Residual of the constraint [dim.: 3x1]
    def e_ij(self, R_ij, R_i, t_i, t_j, t_ij, theta_i, theta_j, theta_ij):
        # Equation (30).
        e_ij = np.zeros([3, 1])
        # 2x1 block
        e_ij[0:2, 0] = np.dot(R_ij.T, np.dot(R_i.T,(t_j - t_i)) - t_ij).reshape(2)
        e_ij[2, 0] = theta_j - theta_i - theta_ij
        return e_ij

    # 2D rotation matrix [dim.: 2x2]
    def R_i(self, theta_i):
        # Equation (31).
        R_i = np.zeros([2, 2])
        R_i[0, 0] = np.cos(theta_i)
        R_i[0, 1] = -np.sin(theta_i)
        R_i[1, 0] = np.sin(theta_i)
        R_i[1, 1] = np.cos(theta_i)
        return R_i

    # Derivate of 2D rotation matrix wrt. theta [dim.: 2x2]
    def dR_i(self, theta_i):
        # Required for equation (32).
        dR_i = np.zeros([2, 2])
        dR_i[0, 0] = -np.sin(theta_i)
        dR_i[0, 1] = -np.cos(theta_i)
        dR_i[1, 0] = np.cos(theta_i)
        dR_i[1, 1] = -np.sin(theta_i)
        return dR_i

    # Derivative of error function wrt. x_i [dim.: 3x3]
    def A_ij(self, R_ij, R_i, dR_i, t_j, t_i):
        # Equation (32).
        # The dimension of A_ij is [num_states x num_states]
        A_ij = np.zeros([3, 3])
        # 2x2 block
        A_ij[0:2, 0:2] = -np.dot(R_ij.T, R_i.T)
        # 2x1 block
        A_ij[0:2, 2] = np.dot(np.dot(R_ij.T, dR_i.T), (t_j-t_i)).reshape(2)
        A_ij[2, 2] = -1.0
        return A_ij

    # Derivative of error function wrt. x_j [dim.: 3x3]
    def B_ij(self, R_ij, R_i):
        # Equation (33).
        # The dimension of B_ij is [num_states x num_states]
        B_ij = np.zeros([3, 3])
        # 2x2 block
        B_ij[0:2, 0:2] = np.dot(R_ij.T, R_i.T)
        B_ij[2, 2] = 1.0
        return B_ij

    # Normalize angles to [-pi,pi).
    def normalizeAngles(self, theta):
        # Iterate through the nodes and normalize the angles.
        for i in range(0, self.num_nodes):
            while theta[i] < -math.pi:
                theta += 2 * math.pi
            while theta[i] >= math.pi:
                theta -= 2 * math.pi
        return theta

    def optimizePoseGraph(self):
        # Maximum number of optimization iterations to avoid getting stuck
        # in infinite while loop.
        max_number_optimization_iterations = 1000
        optimization_iteration_counter = 0
        optimization_error = np.inf
        tolerance = 1.0e-11
        t_i = np.zeros([2, 1])
        t_j = np.zeros([2, 1])
        t_ij = np.zeros([2, 1])
        Omega_ij = np.zeros([3, 3])
        # Make sure we achieve the desired accuracy.
        while optimization_error > tolerance:
            # num_states = 3 * num_nodes (since x,y,theta)
            H = np.zeros([self.num_states, self.num_states])
            b = np.zeros([self.num_states, 1])
            # Iterate over all constraints.
            for constraint in self.constraints:
                # Node i.
                i = int(constraint[0])

                # Node j.
                j = int(constraint[1])

                # Relative translation from node i to node j,
                t_ij[self.index_x] = constraint[2]
                t_ij[self.index_y] = constraint[3]

                # Relative rotation from node i to node j.
                theta_ij = constraint[4]

                # *Global* position of node i (initial guess).
                t_i[self.index_x] = self.x[i, self.index_x]
                t_i[self.index_y] = self.x[i, self.index_y]

                # *Global* position of node j (initial guess).
                t_j[self.index_x] = self.x[j, self.index_x]
                t_j[self.index_y] = self.x[j, self.index_y]

                # *Global* orientation of node i (initial guess).
                theta_i = self.x[i, self.index_theta]

                # *Global* orientation of node j (initial guess).
                theta_j = self.x[j, self.index_theta]

                # Information matrix Omega.
                # First row.
                Omega_ij[0, 0] = constraint[5]
                Omega_ij[0, 1] = constraint[6]
                Omega_ij[0, 2] = constraint[7]
                # Second row.
                Omega_ij[1, 0] = constraint[6]
                Omega_ij[1, 1] = constraint[8]
                Omega_ij[1, 2] = constraint[9]
                # Third row.
                Omega_ij[2, 0] = constraint[7]
                Omega_ij[2, 1] = constraint[9]
                Omega_ij[2, 2] = constraint[10]

                # Compute R_ij, the *local* rotation matrix from node i to node j.
                R_ij = self.R_i(theta_ij)

                # Compute R_i, the *global* orientation of node i.
                R_i = self.R_i(theta_i)

                # Compute R_j, the *global* orientation of node j.
                R_j = self.R_i(theta_j)

                # Compute dR_i, the derivate of R_i wrt. theta_i.
                dR_i = self.dR_i(theta_i)

                # Compute the derivative of the error function wrt. x_i.
                A_ij = self.A_ij(R_ij, R_i, dR_i, t_j, t_i)

                # Compute the derivate of the error function wrt. x_j.
                B_ij = self.B_ij(R_ij, R_i)

                # Compute the residual of the constraint connecting node i and node j
                e_ij = self.e_ij(R_ij, R_i, t_i, t_j, t_ij, theta_i, theta_j, theta_ij)

                # Make sure to get the indices right...
                # i=0: b[0:3]; i=1: b[3:6]; ...
                # j=1: b[3:6]; i=2: b[6:9]; ...
                i_r = 3*i
                i_c = 3*i+3
                j_r = 3*j
                j_c = 3*j+3

                # Compute the coefficient vector.
                # b_i
                b[i_r:i_c] += np.dot(A_ij.T, np.dot(Omega_ij, e_ij)).reshape(3, 1)
                # b_j
                b[j_r:j_c] += np.dot(B_ij.T, np.dot(Omega_ij, e_ij)).reshape(3, 1)

                # Compute the contribution of this constraint to the linear system.
                # H_ii
                H[i_r:i_c,i_r:i_c] += np.dot(A_ij.T, np.dot(Omega_ij, A_ij))
                # H_ij
                H[i_r:i_c,j_r:j_c] += np.dot(A_ij.T, np.dot(Omega_ij, B_ij))
                # H_ji
                H[j_r:j_c,i_r:i_c] += np.dot(B_ij.T, np.dot(Omega_ij, A_ij))
                # H_jj
                H[j_r:j_c,j_r:j_c] += np.dot(B_ij.T, np.dot(Omega_ij, B_ij))

            # Keep the first node fixed.
            H[0:3, 0:3] += np.eye(3, 3)

            # Solve the linear system.
            delta_x = sla.spsolve(H, -b)
            delta_x = delta_x.reshape(self.num_nodes, self.dimensions)

            # Equation (34): Update the states by applying the increments.
            self.x += delta_x

            # Save the current optimization error.
            optimization_error = np.linalg.norm(delta_x, 2)

            # Maximum number of optimization iterations to avoid getting stuck
            # in infinite while loop.
            optimization_iteration_counter += 1
            if optimization_iteration_counter > max_number_optimization_iterations:
                print "WARNING! Reached max. number of iterations before converging to desired tolerance!"
                break

            print "Optimization iter.: ", optimization_iteration_counter, "   optimization error: ", optimization_error

        # The angles are normalized to [-pi,pi) *after* applying the increments.
        self.x[:, self.index_theta] = self.normalizeAngles(self.x[:, self.index_theta])
        return self.x


def main():
    # Relative path to data from exercise sheet.
    base = "../../../pose_graph_optimization/assignment_I_2/"

    # Load the input data.
    vertices = np.genfromtxt(open(base + "vertices.dat"))
    edges = np.genfromtxt(open(base + "edges.dat"))
    lc = np.genfromtxt(open(base + "loop_closures.dat"))

    # Edges and loop-closures are constraints that can be handled the same
    # way in the pose graph optimization backend as remarked in the exercise sheet.
    all_constraints = []
    all_constraints = np.append(edges, lc, axis = 0)

    # Plot the initial values.
    pylab.plot(vertices[:, 1], vertices[:, 2], 'b')
    plt.pause(1)

    # Perform the 2D pose graph optimization according to [1], Algorithm 1, 2D
    pgo = PoseGraphOptimization2D(vertices, all_constraints)
    x_opt = pgo.optimizePoseGraph()

    # Save the optimized states in rows: [x_0, y_0, th_0; x_1, y_1, th_1; ...]
    np.savetxt('results_2D.txt', np.transpose(x_opt))

    # Plot the optimized values.
    pylab.plot(x_opt[:,0], x_opt[:,1], 'g')
    plt.pause(5)

if __name__ == "__main__":
    main()
