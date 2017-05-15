import numpy as np
import random
from scipy import misc
from scipy import ndimage
from sklearn import preprocessing
import matplotlib.pyplot as plt
from PIL import Image
import os

def clustering(X, mu):
    # TODO: Perform optimization of r_nk and fill datapoints into clusters[k] k = 0,...,1-K.
    clusters  = {}
    return clusters
 
def reevaluate_mu(mu, clusters):
    # TODO: Perform optimization of mu_k and return optimized mu_k.
    newmu = []
    return newmu
 
def has_converged(mu, oldmu):
    return set([tuple(j) for j in mu]) == set([tuple(j) for j in oldmu])

def find_centers(X, K):
    # Initialize to K random centers.
    # TODO: Robustify initialization towards global convergence (Exchange initialization of mu).
    mu = random.sample(X, K)
    oldmu = random.sample(X, K)
    
    while not has_converged(mu, oldmu):
        oldmu = mu
        # First step of optimization: Assign all datapoints in X to clusters.
        clusters = clustering(X, mu)
        # Second step of optimization: Optimize location of cluster centers mu.
        mu = reevaluate_mu(oldmu, clusters)
    return(mu, clusters)

# Load precomputed (PCA) features from file.
# WARNING: These are different from the results of the first exercise! Use the provided features file!
features = np.genfromtxt(open("features_k_means.txt"))
# Make sure to normalize your data, you may run into numerical issues otherwise.
features = preprocessing.scale(features)
n_samples, n_features = np.shape(features)
# Initialize centers
initial_mu = random.sample(features, 3)

# Perform Lloyd's algorithm.
mu, clusters = find_centers(features, 3)

# Plot results.
for x in range(len(clusters[0])): plt.plot(clusters[0][x][0], clusters[0][x][1], 'o', markersize=7, color='blue', alpha=0.5, label='Cluster 1')
for x in range(len(clusters[1])): plt.plot(clusters[1][x][0], clusters[1][x][1], 'o', markersize=7, color='red', alpha=0.5, label='Cluster 2')
for x in range(len(clusters[2])): plt.plot(clusters[2][x][0], clusters[2][x][1], 'o', markersize=7, color='green', alpha=0.5, label='Cluster 3')
plt.plot([mu[0][0], mu[1][0], mu[2][0]], [mu[0][1], mu[1][1], mu[2][1]], '*', markersize=20, color='red', alpha=1.0, label='Cluster centers')
plt.xlabel('principal component 1')
plt.ylabel('principal component 2')
plt.title('K-means clustering')
plt.show()

# Save results.
np.savetxt('results_k_means.txt', mu)