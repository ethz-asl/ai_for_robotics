import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt

# Load the dataset of orb features from file.
orb_features = np.genfromtxt(open("orb_features.txt"))
orb_features = orb_features.T

orb_size = len(orb_features[:, 0])
# TODO: Compute covariance (exchange the value for None).
cov = None

# TODO: Compute eigenvectors and eigenvalues (exchange the value for None).
eig_val_cov = None
eig_vec_cov = None

# Sort eigenvectors and corresponding eigenvalues in descending order.
eig_pairs = [(np.abs(eig_val_cov[i]), eig_vec_cov[:, i]) for i in range(len(eig_val_cov))]
eig_pairs.sort(key=lambda x: x[0], reverse=True)
    
# TODO: Compute 5 dimensional feature vector based on largest eigenvalues and normalize the output (exchange the value for None).
pca_features = None

# Normalize pca features.
pca_features = preprocessing.scale(pca_features)

# 2D plot of first 2 principal components.
plt.scatter(pca_features[:, 0], pca_features[:, 1], marker = 'o')
plt.xlabel('principal component 1')
plt.ylabel('principal component 2')
plt.title('PCA result')
plt.show()

# Save results.
np.savetxt('results_pca.txt', pca_features)
