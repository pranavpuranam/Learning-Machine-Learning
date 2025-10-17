# TUTORIAL SHEET 1 - MY CODE

# imports

from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

# random seed

np.random.seed(0)

# (1) Generate and plot a test dataset -----------------------------------------------------------------------------

# (a) producing a dataset
X, y = datasets.make_classification(n_samples = 100, n_features = 2, n_informative = 2, n_redundant = 0)

# scaling + shifting data
X[:, 0] = np.abs(X[:, 0] * 0.5 + 5)
X[:, 1] = np.abs(X[:, 1] * 30 + 160)

# set up axes for a plot
fig, ax = plt.subplots()

# (b) plot data
ax.scatter(X[y==0, 0],X[y==0, 1])
ax.scatter(X[y==1, 0], X[y==1, 1])

# define axis limits
ax.set_xlim(3, 8)
ax.set_ylim(0, 250)

# add a line of separation
x_separation = np.linspace(X[:,0].min(), X[:,0].max(), 100)
y_separation = -275 * x_separation + 1475
plt.plot(x_separation, y_separation)
plt.show()

# (2) Make a function to generate a suitable covariance matrix -----------------------------------------------------

# (a) function to generate rotated covariance matrix
def get_cov(sdx = 1, sdy = 1, rotangdeg = 0):
    covar = np.array([[sdx**2, 0], [0, sdy**2]])
    rot_ang = rotangdeg / 360 * 2 * np.pi
    rot_mat = np.array([[np.cos(rot_ang), -np.sin(rot_ang)], [np.sin(rot_ang), np.cos(rot_ang)]])
    covar = np.matmul(np.matmul(rot_mat, covar), rot_mat.T)
    return covar

# (b) function to generate a grid of points
def gen_sample_grid(npx=200, npy=200, limit=1):
    x1line = np.linspace(-limit, limit, npx)
    x2line = np.linspace(-limit, limit, npy)
    x1grid, x2grid = np.meshgrid(x1line, x2line)
    Xgrid = np.array([x1grid, x2grid]).reshape([2,npx*npy]).T
    return Xgrid,x1line,x2line

# (c) generate covariance matrix with std devs 1 and 0.3, rotated by 30 degrees
covar = get_cov(sdx = 1, sdy = 0.3, rotangdeg = 30)

# (d) generate sample grid
Xgrid, x1line, x2line = gen_sample_grid(npx = 200, npy = 200, limit = 1)

# (e) compute the multivariate Gaussian PDF for all grid points
inv_covar = np.linalg.inv(covar)
det_covar = np.linalg.det(covar)
p = 1 / (2 * np.pi * np.sqrt(np.linalg.det(covar))) * np.exp(-1 / 2 * (np.matmul(Xgrid, np.linalg.inv(covar)) * Xgrid).sum(-1))

pdf_grid = p.reshape(200,200)

plt.figure(figsize=(6,6))
plt.contourf(x1line, x2line, pdf_grid, levels=50, cmap='viridis')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('2D Gaussian PDF')
plt.colorbar(label='Probability density')
plt.show()

samples = np.random.multivariate_normal(mean=[0,0], cov=covar, size=100)

plt.figure(figsize=(6,6))
plt.contourf(x1line, x2line, pdf_grid, levels=50, cmap='viridis')
plt.scatter(samples[:,0], samples[:,1], color='red', s=20, label='Samples')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('2D Gaussian PDF with Samples')
plt.xlim([-1,1])
plt.ylim([-1,1])
plt.legend()
plt.show()

# seems like a pretty good representation