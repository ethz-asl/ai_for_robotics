from scipy.optimize import minimize
import numpy as np
import pylab as pl
from mpl_toolkits.mplot3d import Axes3D
import math


def f(x):
  """ Function that returns x_0^2 + e^{0.5*x_0} + 10*sin(x_1) + x_1^2. """
  return x[0] ** 2 + math.exp(0.5 * x[0]) + 10 * math.sin(x[1]) + x[1] ** 2


def fprime(x):
  """ The derivative of f. """
  ddx0 = 2 * x[0] + 0.5 * math.exp(0.5 * x[0])
  ddx1 = 10 * math.cos(x[1]) + 2 * x[1]
  return np.array([ddx0, ddx1])


opt_out = minimize(f, x0=np.array(
    [10, 10]), jac=fprime, tol=1e-8, method='BFGS', options={'disp': True})

# Plotting
pl.close('all')
r = 6
x_range = np.linspace(-r, r)
y_range = np.linspace(-r, r)
X, Y = np.meshgrid(x_range, y_range)
Z = np.zeros(X.shape)
for i in range(X.shape[0]):
  for j in range(X.shape[1]):
    Z[i, j] = f(np.array([X[i, j], Y[i, j]]))

fig = pl.figure('Cost function')
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap=pl.cm.coolwarm, alpha=0.6)
ax.scatter(opt_out.x[0], opt_out.x[1], f(opt_out.x), c='r', s=50)

pl.show(block=False)
